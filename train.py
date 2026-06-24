import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import time
from datetime import timedelta
import logging
from tqdm import tqdm
from pathlib import Path
import random
import numpy as np
from collections import Counter

import torch
from torch import nn, distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy

from utils.misc import create_logger
dir_bakcup = str(Path(__file__).parent)

from model.flm_occ import FLMOcc
from model import SQMM
from utils.training import \
    create_lr_scheduler, seed_everything, backup_codes, get_total_norm, check_grad_nan_inf
from utils.metrics import \
    pre_recall_f1score_iou, stat_pre_recall_f1score_iou, MeanIoUAcc, generate_eval_tables
    
from datasets_.scannet import ScannetDataset as UsedDataset
import configs.scannet as UsedConfig
DC, MC, TC = UsedConfig.Dataset, UsedConfig.Model, UsedConfig.Train

if TC.use_swanlab:
    import swanlab



def ce_loss(logits, target, cls_weights=None):
    return F.cross_entropy(
        logits, 
        target,
        weight=cls_weights,
        reduction='mean'
    )


def sqmm_loss(density, scale, e1e2):
    # density part
    loss = density.clamp(1e-8).log().mean(-1).neg()
    
    # volume part
    loss += SQMM.kernel_volume_sq_log(scale, e1e2).exp().sum(-1).clamp(1e-8).log()

    return loss


def gmm_loss(likelihood, gs_scales):
    # likelihood part
    loss = likelihood.clamp(1e-8).log().mean(-1).neg()
    
    # volume part
    loss += gs_scales.prod(-1).sum(-1).clamp(1e-8).log()
    
    return loss # [..., levels]


def likelihood_reg(likelihood):
    return (likelihood - 1).square().mean(-1)


def cal_losses(outputs, batch, TC):
    (
        tuple_LBGC,
        B_LPdensity,
        B_LPlogit,
        B_xyz_nempty,
        B_labels_nempty,
        B_cls_weights
    ) = (
        outputs['tuple_LBGC'],
        outputs['densities_sparse'],
        outputs['logits_sparse'],
        batch['xyz_nempty'],
        batch['labels_nempty'],
        batch['cls_weights']
    )
    
    levels_out = TC.levels_out
    num_levels = TC.num_levels
    dd = tuple_LBGC[0] # device, dtype
    gmm   = torch.tensor([0.]*num_levels).to(dd)
    lreg  = torch.tensor([0.]*num_levels).to(dd)
    ce    = torch.tensor(0.).to(dd)

    weights_levels_out = torch.tensor(TC.weights_gmm_level).to(dd)
    
    batch_size = len(B_LPdensity)
    for i in range(batch_size):
        # GMM term
        Pdensity = B_LPdensity[i]#[levels_out]
        scales = tuple_LBGC[1][levels_out, i]
        # # Superquadric kernel
        if MC.use_SQMM:
            e1e2 = tuple_LBGC[-2][levels_out, i]
            gmm[levels_out] += sqmm_loss(Pdensity, scales, e1e2) * weights_levels_out
        else:
        # # Gaussian kernel
            gmm[levels_out] += gmm_loss(Pdensity, scales) * weights_levels_out
        
        # regularization term
        # import pdb; pdb.set_trace()
        lreg[levels_out] += (
            likelihood_reg(Pdensity) * weights_levels_out
            # if TC.lambda_lreg > 0 else 0.
        )
        
        # cross entropy term
        logits = B_LPlogit[i][-1]
        target = B_labels_nempty[i]
        ce += ce_loss(logits, target)
        
    _gmm   = gmm.detach().clone()
    _lreg  = lreg.detach().clone()
    _ce    = ce.detach().clone()
    
    reduce = lambda x: dist.all_reduce(x, dist.ReduceOp.AVG) if dist.is_initialized() else x
    reduce(_gmm )
    reduce(_lreg)
    reduce(_ce  )
    
    mean_ = lambda x: x.item() / batch_size
    dict_loss = {
        'GMM':   mean_(_gmm [-1]),
        'Lreg':  mean_(_lreg[-1]),
        'CE':    mean_(_ce   ),
    }
    
    gmm   = gmm.sum()  * (TC.lambda_gmm if TC.lambda_gmm > 0 else 0.)
    lreg  = lreg.sum() * (TC.lambda_lreg if TC.lambda_lreg > 0 else 0.)
    ce    = ce         * (TC.lambda_ce if TC.lambda_ce > 0 else 0.)
    
    loss_train = (gmm + lreg + ce) / batch_size
    
    return loss_train, dict_loss


def cal_metrics(outputs, batch, include_empty):
    (
        B_Pdensity,
        B_Plogit,
        list_map_out__in,
        list_mask_input,
    ) = (
        outputs['densities_dense'],
        outputs['logits_dense'],
        outputs['map_out__in'],
        outputs['list_mask_input'],
    )
    
    list_pre = []
    list_rec = []
    list_f1s = []
    list_iou = []
    list_valid_query = []  # ratio of query points that have supervision
    list_occ_in_loss = []  # ratio of occupied points that have supervision
    
    for i in range(len(B_Pdensity)):
        Pdensity  = B_Pdensity[i]
        Plogit  = B_Plogit[i]
        map_out__in = list_map_out__in[i]
        mask_input  = list_mask_input[i]
        labels = batch['labels_query'][i] + int(not include_empty)
        label_used = labels[map_out__in][mask_input]
        
        gt = label_used.bool()
        pred = Pdensity > 0.5
        
        # find labels that are not used in loss
        label_not_used = labels[map_out__in][~mask_input].bool()
        
        gt = torch.cat([gt, label_not_used])
        pred = torch.cat([pred, torch.zeros_like(label_not_used)])
        
        pre_recall_f1score_iou(
            gt,
            pred,
            list_pre,
            list_rec,
            list_f1s,
            list_iou,
        )
        
        list_valid_query.append(mask_input.sum().item()/mask_input.shape[0])
        
        num_nempty = (labels>0).sum().item()
        ratio = num_nempty / labels.shape[0] if num_nempty > 0 else 0
        list_occ_in_loss.append(ratio)

    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=pred.device)
    t_pre = tensor(list_pre)
    t_rec = tensor(list_rec)
    t_f1s = tensor(list_f1s)
    t_iou = tensor(list_iou)
    t_valid_query = tensor(list_valid_query)
    t_occ_in_loss = tensor(list_occ_in_loss)

    return (
        fabric.all_gather(t_pre).flatten().tolist(),
        fabric.all_gather(t_rec).flatten().tolist(),
        fabric.all_gather(t_f1s).flatten().tolist(),
        fabric.all_gather(t_iou).flatten().tolist(),
        fabric.all_gather(t_valid_query).flatten().tolist(),
        fabric.all_gather(t_occ_in_loss).flatten().tolist(),
    )


def accumulate_confusion_matrix(iou_cal, miou_cal, outputs, batch, include_empty: bool=True):
    (
        B_Pdensity,
        B_Plogit,
        list_map_out__in,
        list_mask_input,
    ) = (
        outputs['densities_dense'],
        outputs['logits_dense'],
        outputs['map_out__in'],
        outputs['list_mask_input'],
    )
    
    for i in range(len(B_Pdensity)):
        Pdensity = B_Pdensity[i]
        Plogit   = B_Plogit[i]
        map_out__in = list_map_out__in[i]
        mask_input  = list_mask_input[i]
        labels = batch['labels_query'][i] + int(not include_empty)

        # label of empty is -1 if not include_empty
        gt = labels[map_out__in][mask_input]
        pred_geo = Pdensity > 0.5 # non empty is 1, empty is 0
        
        # find labels that are not covered by GS
        label_not_covered = labels[map_out__in][~mask_input]
        
        gt = torch.cat([gt, label_not_covered])
        pred_geo = torch.cat([pred_geo.long(), torch.zeros_like(label_not_covered)])
        
        pred_sem = torch.softmax(Plogit, dim=-1).argmax(dim=-1).long() + int(not include_empty)
        pred_sem = torch.cat([pred_sem, torch.zeros_like(label_not_covered)])
        pred_sem = pred_sem * pred_geo
        
        if iou_cal is not None:
            iou_cal.update(pred_geo, gt.bool())
        if iou_cal is not None:
            miou_cal.update(pred_sem, gt)
    
    return iou_cal, miou_cal


def validate_model(model, dataloader, fabric, pbar=None):
    all_list_pre = []
    all_list_rec = []
    all_list_f1s = []
    all_list_iou = []
    all_list_valid_query = []  # ratio of query points that have supervision
    all_list_occ_in_loss = []  # ratio of occupied points that have supervision
    
    iou_cal = MeanIoUAcc(2, ignore_index=-1, device=fabric.device)
    miou_cal = MeanIoUAcc(DC.num_semantics, ignore_index=-1, device=fabric.device)
    
    num_batch = len(dataloader)
    
    loss = 0.
    dict_loss = Counter()
    
    for i, batch in enumerate(dataloader):
        outputs = model(batch, do_local_agg=True, levels_out=TC.levels_out)
        (
            list_pre,
            list_rec,
            list_f1s,
            list_iou,
            list_valid_query,
            list_occ_in_loss,
        ) = cal_metrics(
            outputs,
            batch,
            DC.include_empty,
        )
        
        all_list_pre.extend(list_pre)
        all_list_rec.extend(list_rec)
        all_list_f1s.extend(list_f1s)
        all_list_iou.extend(list_iou)
        all_list_valid_query.extend(list_valid_query)
        all_list_occ_in_loss.extend(list_occ_in_loss)
        if pbar is not None:
            pbar.set_description(
                f'Epoch: {i_epoch+1:2d}/{total_epochs}, '
                + f'Iter: {i_train+iter_offset+i_epoch*iters_per_epoch+1:3d}/{total_iters}, '
                + f'Val: {i+1:3d}/{len(dataloader)}, '
            )
        
        accumulate_confusion_matrix(iou_cal, miou_cal, outputs, batch, DC.include_empty)
        
        batch_loss, batch_dict_loss = cal_losses(outputs, batch, TC)
        loss += batch_loss.item()
        dict_loss.update(batch_dict_loss)
    
    loss /= num_batch
    for k, v in dict_loss.items():
        dict_loss[k] = v / num_batch
        
    iou_bin = iou_cal.sync().compute()[1]
    miou, ious = miou_cal.sync().compute()
    
    tensor = lambda x: torch.tensor(x, dtype=torch.float32, device=fabric.device)
    t_pre = tensor(all_list_pre)
    t_rec = tensor(all_list_rec)
    t_f1s = tensor(all_list_f1s)
    t_iou = tensor(all_list_iou)
    t_valid_query = tensor(all_list_valid_query)
    t_occ_in_loss = tensor(all_list_occ_in_loss)
    
    # broadcast results across all gpus
    return (
        fabric.all_gather(t_pre).flatten().tolist(),
        fabric.all_gather(t_rec).flatten().tolist(),
        fabric.all_gather(t_f1s).flatten().tolist(),
        fabric.all_gather(t_iou).flatten().tolist(),
        fabric.all_gather(t_valid_query).flatten().tolist(),
        fabric.all_gather(t_occ_in_loss).flatten().tolist(),
        iou_bin, miou, ious,
        loss, dict_loss
    )



if __name__ == "__main__":
    # region System
    torch.set_float32_matmul_precision('high')
    if TC.mixed_precision:
        fabric = Fabric(accelerator="cuda", devices=TC.devices, precision="bf16-mixed",
                        # strategy='ddp')
                        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=100)))
    else:
        fabric = Fabric(accelerator="cuda", devices=TC.devices, 
                        # strategy='ddp')
                        strategy=DDPStrategy(find_unused_parameters=False, timeout=timedelta(seconds=100)))
    fabric.launch()
    
    if TC.check_anomaly:
        torch.autograd.set_detect_anomaly(True)
    
    printi = lambda x: None
    printd = lambda x: None
    if fabric.is_global_zero:
        os.makedirs(TC.exp_dir, exist_ok=True)
        path_log = os.path.join(TC.exp_dir, f'flm-occ-{time.strftime("%Y%m%d_%H%M%S")}.log')
        logger = create_logger('flm-occ', logging.INFO, path_log)
        if os.path.islink('flm-occ.log'):
            os.remove('flm-occ.log')
        os.symlink(path_log, 'flm-occ.log')
        printi = logger.info
        printd = logger.debug
        printi(TC.notes)
        printi(TC.exp_dir)
    
    # control reproductivity # May not work!
    if TC.random_seed > 0:
        seed_everything(TC.random_seed)
        
    # region Model and Optimizer
    # ============ Setup Model and Optimizer ============
    # model setup
    model_training = FLMOcc(MC, fabric.global_rank)#.to(fabric.device) # necessary although official doc says not    
    
    for n, p in model_training.named_parameters():
        if p.requires_grad:
            printd(n)
    
    model_training.init_weights()

    # optimizer setup
    def params_filtered_by_name(named_params, keywords='', exclude=False):
        if isinstance(keywords, str):
            keywords = [keywords]

        def condition(np):
            name = np[0]
            return np[1].requires_grad and any(k in name for k in keywords)
        
        if exclude:
            def condition(np):
                name = np[0]
                return np[1].requires_grad and all(k not in name for k in keywords)
        
        return [p for n, p in filter(condition, named_params)]
    
    l = [
        {
            'params': params_filtered_by_name(model_training.named_parameters(), 'rgb_encoder.pretrained', exclude=True),
            'lr': TC.lr,
            'weight_decay': TC.weight_decay,
            'betas': TC.betas,
        },
        {
            'params': params_filtered_by_name(model_training.named_parameters(), 'rgb_encoder.pretrained'),
            'lr': TC.pretrained_lr,
            'weight_decay': TC.weight_decay,
            'betas': TC.betas,
        },
    ]
    optimizer = torch.optim.AdamW(l)
    optimizer.zero_grad()

    # distribute model and optimizer
    model_training = nn.SyncBatchNorm.convert_sync_batchnorm(model_training)#.to(fabric.device)
    model_training, optimizer = fabric.setup(model_training, optimizer)

    # region Dataset
    # ============ Dataset =============
    ## training set
    dataset = UsedDataset(DC, 'train', fabric.global_rank)
    sampler = torch.utils.data.DistributedSampler(dataset) # 自动设置world_size和rank
    dataloader = DataLoader(
        dataset, 
        batch_size=TC.batch_size,
        sampler=sampler,
        # shuffle=True,
        num_workers=TC.num_workers,
        persistent_workers=TC.persistent_workers,
        pin_memory=True,
        prefetch_factor=TC.prefetch_factor,
        collate_fn=UsedDataset.collate_fn,
        generator=torch.Generator().manual_seed(TC.random_seed),
    )

    # region scheduler
    # learning rate schedule, multiply a factor with the learning_rate
    iters_per_epoch = len(dataloader)
    warmup_epochs = TC.warmup_iters / iters_per_epoch
    cosine_epochs = TC.cosine_iters / iters_per_epoch
    if TC.use_scheduler:
        lr_scheduler = create_lr_scheduler(
            optimizer,
            stype=TC.stype,
            steps_per_epoch=iters_per_epoch,
            warmup_epochs=warmup_epochs,
            cosine_epochs=cosine_epochs,
            f_min=TC.f_min,
            f_max=TC.f_max,
        )

    # region Checkpoint
    if TC.training_ckpt and not os.path.exists(TC.training_ckpt):
        printi('Ckpt path is wrong!')
        raise FileNotFoundError
    
    current_iter = torch.tensor([0], dtype=torch.long, device=fabric.device)
    if TC.training_ckpt:
        # Resuming from checkpoint
        if fabric.is_global_zero:
            printi(f'Loading checkpoint {TC.training_ckpt}')
        path_ckpt = TC.training_ckpt
        checkpoint = fabric.load(path_ckpt, weights_only=False)
        
        # model
        model_training.load_state_dict(checkpoint["model_state_dict"], strict=True)
        printi('Loaded model')
        
        # optimizer
        # if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
        printi('Loaded optimizer')

        if TC.use_scheduler:
            lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            printi('Loaded scheduler')
            
        # iteration
        if TC.finetune:
            assert TC.training_ckpt is not None, "Please provide a pretrained checkpoint for finetuning"
            current_iter = torch.tensor([0], dtype=torch.long).to(fabric.device)
        else:
            current_iter = torch.tensor(checkpoint["iteration"], dtype=torch.long, device=fabric.device)

        # RNG
        # if 'rng_state' in checkpoint:
        rng_state = checkpoint['rng_state']
        random.setstate(rng_state['python_rng_state'])
        np.random.set_state(rng_state['numpy_rng_state'])
        torch.set_rng_state(rng_state['torch_rng_state'])
        torch.cuda.set_rng_state_all(rng_state['torch_cuda_rng_state'])
        printi('Loaded RNG state')

        for k, v in checkpoint.items():
            del v
        del checkpoint
        torch.cuda.empty_cache()
    
    current_iter = current_iter.item() # Does it need to be broadcast?
    
    # region Dataset
    # ============ Dataset =============
    ## validation set
    dataset_val = UsedDataset(DC, 'val', fabric.global_rank)
    sampler_val = torch.utils.data.DistributedSampler(dataset_val, shuffle=False) # 自动设置world_size和rank
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=TC.batch_size,
        sampler=sampler_val,
        num_workers=TC.num_workers,
        persistent_workers=TC.persistent_workers,
        pin_memory=True,
        prefetch_factor=TC.prefetch_factor,
        collate_fn=UsedDataset.collate_fn,
        generator=torch.Generator().manual_seed(TC.random_seed),
    )
    
    if fabric.is_global_zero:
        printi('-'*30+f'{len(dataset)} samples loaded'+'-'*30)
        printi('-'*30+f'{len(dataset_val)} samples loaded'+'-'*30)
    
    # distribute datasets
    dataloader = fabric.setup_dataloaders(dataloader, use_distributed_sampler=False) # false so that each dataset use part of the full dataset
    dataloader_val = fabric.setup_dataloaders(dataloader_val, use_distributed_sampler=False) # False to use my own sampler


    # region SWANLAB
    dir_log = TC.exp_dir
    if fabric.is_global_zero:
        timestampt = backup_codes(dir_bakcup, TC.exp_dir)
        
        flag_swanlab = False
        def setup_swanlab():
            swanlab.init(
                # 设置项目名
                project=TC.swanlab_proj,
                workspace='gcchen',
                resume='allow',
                id=TC.swanlab_id,
                notes=TC.notes,
                experiment_name=os.path.basename(TC.exp_dir),
                # 设置超参数
                config={
                    "learning_rate": TC.lr,
                    "architecture": "DAv2_EmbodiedOcc",
                    "dataset": UsedDataset.__name__,
                    "max iters": TC.total_iters,
                    "batch_size": TC.batch_size*fabric.world_size,
                    "num_gs": MC.num_gs,
                    "num_refine": MC.num_refine_blocks,
                    "mixture model": 'SQquadric' if MC.use_SQMM else 'Gaussian',
                }
            )
        
    
    # region Training
    # ============ Training loop ============
    size_val = len(dataloader_val)
    total_iters = TC.total_iters
    total_epochs = int(total_iters // iters_per_epoch + 1)
    current_epoch = (current_iter+1) // iters_per_epoch
    iter_offset = (current_iter+1) % iters_per_epoch if current_iter > 0 else 0
    if fabric.is_global_zero:
        printi(f'{current_iter=} {total_iters=}, {current_epoch=}, {iter_offset=}, {iters_per_epoch=}, {size_val=}')
    
    if fabric.is_global_zero:
        printd('\n' + MC.tostring())
        printd('\n' + DC.tostring())
        printd('\n' + TC.tostring())
    
    # region Metrics
    iou_cal  = MeanIoUAcc(2, ignore_index=-1, device=fabric.device)
    miou_cal = MeanIoUAcc(DC.num_semantics, ignore_index=-1, device=fabric.device)
    
    model_training.train()
    if fabric.is_global_zero:
        pbar = tqdm(total=iters_per_epoch, ncols=100)
    else:
        pbar = None
    for i_epoch in range(current_epoch, total_epochs):
        sampler.set_epoch(i_epoch)        
        if i_epoch < current_epoch:
            continue
            
        for i_train, batch in enumerate(dataloader):
            if i_train + iter_offset + 1 == iters_per_epoch and iter_offset > 0:
                break
            is_last_iter = (current_iter + 1) == total_iters
            is_end_of_epoch = (current_iter + 1) % iters_per_epoch == 0
            
            # region Batch Info
            # ============ Batch Info ============
            # idxs = batch['idxs']
            # str_ = 'Training samples:\n'
            # str_ += f'{idxs}\n'
            # str_ += '\n'.join([dataset.list_path_rgb[i] for i in idxs])
            # printd(str_)
            
            # region Inference and Loss
            # ============  Inference and Loss   ============
            flag_log = is_last_iter or (
                # is_end_of_epoch and iters_per_epoch > TC.log_iters # avoid redundant logging
                False # avoid redundant logging
            ) or (
                TC.log_iters > 0 and ((current_iter + 1) % TC.log_iters) == 0
            )
            if flag_log:
                torch.cuda.empty_cache()
            outputs = model_training(batch, do_local_agg=flag_log, levels_out=TC.levels_out)
            
            loss_train, dict_loss = cal_losses(outputs, batch, TC)
            loss_dummy = 0. # used when there are some modules that do not have gradients
            loss_bwd = loss_train + loss_dummy
            
            loss_train_log = loss_train.detach().clone()
            loss_train_log = fabric.all_reduce(loss_train_log, reduce_op='mean').item()
            if flag_log:
                (
                    list_pre,
                    list_rec,
                    list_f1s,
                    list_iou,
                    list_valid_query,
                    list_occ_in_loss,
                ) = cal_metrics(
                    outputs,
                    batch,
                    DC.include_empty,
                )
                
                dict_stat = stat_pre_recall_f1score_iou(
                    list_pre,
                    list_rec,
                    list_f1s,
                    list_iou,
                    list_valid_query,
                    list_occ_in_loss,
                    hist_bins=5,
                )
                
                iou_cal.reset()
                miou_cal.reset()
                accumulate_confusion_matrix(iou_cal, miou_cal, outputs, batch, DC.include_empty)
                iou_cal.sync()
                miou_cal.sync()
                iou_bin = iou_cal.compute()[1]
                miou, ious = miou_cal.compute()
                
                if fabric.is_global_zero:
                    printd(
                        '\n'+'='*20+f' Iter {current_iter:4d} ' + '='*20
                        + '\n' + f'Loss: {loss_train_log:.3f}'
                        + '\n'+'Training results:'
                        + '\n'+dict_stat['str_valid_query']
                        + '\n'+dict_stat['str_occ_in_loss']
                        + '\n'+dict_stat['str_pre']
                        + '\n'+dict_stat['str_rec']
                        + '\n'+dict_stat['str_f1s']
                        + '\n'+dict_stat['str_iou']
                    )

                    if TC.use_swanlab:
                        if not flag_swanlab:
                            setup_swanlab()
                            flag_swanlab = True
                        swanlab.log({
                            "train/loss": loss_train_log,
                            "train/gmm": dict_loss['GMM'],
                            "train/lreg": dict_loss['Lreg'],
                            "train/ce": dict_loss['CE'],
                            "IoU_train/Geo": iou_bin[1].item(),
                            "IoU_train/nempty_mIoU": ious[1:].nanmean().item(),
                            **{f"IoU_train/{UsedDataset.namemap[i]}": iou_i.item()
                               for i, iou_i in enumerate(ious)},
                        }, step=current_iter)
                        
                        # _str, md_str, tex_str = generate_eval_tables(ious, UsedDataset.namemap, iou_bin)
                        # swanlab.log({"tables/train-IoUs": swanlab.Text(_str)}, step=current_iter)
                        # swanlab.log({"tables/train-IoUs-md": swanlab.Text(md_str)}, step=current_iter)
                        # printd('\n'+tex_str)          
            
            # region Optimization 
            # ============  Optimization  ============
            if loss_bwd.isnan() or loss_bwd.isinf():
                printd(f"⚠️ Iter {current_iter:4d}: Found NaN loss!")
                current_iter += 1
                if fabric.is_global_zero:
                    pbar.update(1)
                continue

            fabric.backward(loss_bwd)
            
            if TC.check_grad_nan_inf:
                if check_grad_nan_inf(model_training):
                    printd(f"✅ Iter {current_iter:4d}: Gradient is valid.")
                else:
                    printd(f"⚠️ Iter {current_iter:4d}: Found NaN or Inf in gradients, skipping this step.")
                    optimizer.zero_grad() # dummy gradient step
                    current_iter += 1
                    if fabric.is_global_zero:
                        pbar.update(1)
                    
                    continue
            
            if TC.grad_norm > 0:
                norm_grad = torch.nn.utils.clip_grad_norm_(model_training.parameters(), max_norm=TC.grad_norm)
            else:
                if flag_log:
                    del outputs, loss_bwd, loss_train
                    # torch.cuda.empty_cache()
                    with torch.no_grad():
                        norm_grad = get_total_norm(model_training.parameters(), norm_type=2).item()
            if fabric.is_global_zero and (
                ((current_iter + 1) % TC.log_iters == 0)
                or (flag_log and fabric.is_global_zero)
            ):
                printd(f"   Iter {current_iter:4d}: Grad norm {norm_grad:.4f}")
                if TC.use_swanlab:
                    swanlab.log({"gradients/norm": norm_grad}, step=current_iter)
                    swanlab.log({"gradients/lr": optimizer.param_groups[0]['lr']}, step=current_iter)

            optimizer.step()
            optimizer.zero_grad()
            if TC.use_scheduler:
                lr_scheduler.step()
            
            # region tqdm
            # ============ tqdm ===============
            if fabric.is_global_zero:
                pbar.set_description(
                    f'Epoch: {i_epoch+1:2d}/{total_epochs}, '
                    +f'Iter: {i_train+iter_offset+i_epoch*iters_per_epoch+1:3d}/{total_iters}, '
                    +f'Loss: {loss_train_log:.3f} '
                )
            
            # region Checkpoint
            # ============ Model saving =================
            if is_last_iter or (
                TC.ckpt_iters > 0 and (current_iter + 1) % TC.ckpt_iters == 0
            ):
                # Do not add fabric.is_global_zero here. 
                # Although all processes run save, only rank 0 actually does IO.
                if fabric.is_global_zero:
                    pbar.set_description(
                        f'Epoch: {i_epoch+1:2d}/{total_epochs}, '
                        +f'Iter: {i_train+iter_offset+i_epoch*iters_per_epoch+1:3d}/{total_iters}, '
                        +f'Loss: {loss_train_log:.3f} '
                        +'Saving checkpoint'
                    )
                ckpt_save_dict = {
                    "model_state_dict": model_training,
                    "optimizer_state_dict": optimizer,
                    "iteration": current_iter,
                    "loss": loss_train_log,
                    'rng_state': {
                        'python_rng_state': random.getstate(),
                        'numpy_rng_state': np.random.get_state(),
                        'torch_rng_state': torch.get_rng_state(),
                        'torch_cuda_rng_state': torch.cuda.get_rng_state_all(),
                    },
                }
                if TC.use_scheduler:
                    ckpt_save_dict["scheduler_state_dict"] = lr_scheduler
                
                path_ckpt = f'{dir_log}/ckpt/{current_iter:06d}.pth'
                if TC.training_ckpt is None or not os.path.exists(path_ckpt):
                    if os.path.exists(path_ckpt) and fabric.is_global_zero:
                        printd('⚠️ Checkpoint already exists, overwriting it!')
                    fabric.save(path_ckpt, ckpt_save_dict)

                
            # region Validation
            # ========== Validation =============
            if is_last_iter or (
                TC.val_iters > 0 and (current_iter + 1) % TC.val_iters == 0 and 
                ( 
                    iter_offset == 0 or
                    (iter_offset > 0 and i_train > 0)
                )
            ):
                with torch.no_grad():
                    (
                        list_pre,
                        list_rec,
                        list_f1s,
                        list_iou,
                        list_valid_query,
                        list_occ_in_loss,
                        iou_bin, miou, ious,
                        val_loss, val_dict_loss
                    ) = validate_model(model_training.eval(), dataloader_val, fabric, pbar)
                model_training.train()
                # DAv2
                # model_training.rgb_encoder.pretrained.eval()
                
                dict_stat = stat_pre_recall_f1score_iou(
                    list_pre,
                    list_rec,
                    list_f1s,
                    list_iou,
                    list_valid_query,
                    list_occ_in_loss,
                    hist_bins=5,
                )

                if fabric.is_global_zero:
                    printd(
                        '\n'+'Validation results:'
                        + '\n'+dict_stat['str_valid_query']
                        + '\n'+dict_stat['str_occ_in_loss']
                        + '\n'+dict_stat['str_pre']
                        + '\n'+dict_stat['str_rec']
                        + '\n'+dict_stat['str_f1s']
                        + '\n'+dict_stat['str_iou']
                    )

                    if TC.use_swanlab:
                        swanlab.log({
                            "eval/loss":  val_loss,
                            "eval/gmm":   val_dict_loss['GMM'],
                            "eval/lreg":  val_dict_loss['Lreg'],
                            "eval/ce":    val_dict_loss['CE'],
                            "IoU/Geo": iou_bin[1].item(),
                            "IoU/nempty_mIoU": ious[1:].nanmean().item(),
                            **{f"IoU/{UsedDataset.namemap[i]}": iou_i.item()
                               for i, iou_i in enumerate(ious)},
                        }, step=current_iter)
                        
                        _str, md_str, tex_str = generate_eval_tables(ious, UsedDataset.namemap, iou_bin)
                        swanlab.log({"tables/eval-IoUs": swanlab.Text(_str)}, step=current_iter)
                        swanlab.log({"tables/eval-IoUs-md": swanlab.Text(md_str)}, step=current_iter)
                        printd('\n'+tex_str)

            # ============ End of Validation =============

            # ============ End of Iteration  =============
            if fabric.is_global_zero:
                # printd(f"   Iter {current_iter:4d}: Finished.")
                pbar.update(1)
            current_iter += 1
            
            if is_last_iter:
                break
            
        if is_last_iter:
            break
            
        # ============ End of epoch =============
        iter_offset = 0
        if fabric.is_global_zero:
            pbar.reset()

    swanlab.finish() if TC.use_swanlab and fabric.is_global_zero else None
    # ============ End of training =============
