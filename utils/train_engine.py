# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os, torch, datetime, shutil, time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from openvqa.models.model_loader import ModelLoader
from openvqa.utils.optim import get_optim, adjust_lr
from utils.test_engine import test_engine, ckpt_proc


def _auto_tune_loader_by_vram(__C):
    if not bool(getattr(__C, 'AUTO_TUNE_LOADER_BY_VRAM', False)):
        return
    if not torch.cuda.is_available():
        print('[AUTO_TUNE] CUDA not available, skip auto tuning.')
        return

    total_gb = torch.cuda.get_device_properties(0).total_memory / float(1024 ** 3)
    rules = getattr(__C, 'AUTO_TUNE_LOADER_RULES', [])
    if not isinstance(rules, list) or len(rules) == 0:
        print('[AUTO_TUNE] AUTO_TUNE_LOADER_RULES is empty, skip auto tuning.')
        return

    valid_rules = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if 'min_gb' not in rule or 'batch_size' not in rule:
            continue
        valid_rules.append(rule)
    if len(valid_rules) == 0:
        print('[AUTO_TUNE] No valid rules found, skip auto tuning.')
        return

    valid_rules.sort(key=lambda x: float(x.get('min_gb', 0.0)), reverse=True)
    selected = valid_rules[-1]
    for rule in valid_rules:
        if total_gb >= float(rule.get('min_gb', 0.0)):
            selected = rule
            break

    old_bs = int(__C.BATCH_SIZE)
    old_nw = int(__C.NUM_WORKERS)
    old_pf = int(getattr(__C, 'PREFETCH_FACTOR', 4))

    bs_target = int(selected.get('batch_size', old_bs))
    nw_target = int(selected.get('num_workers', old_nw))
    pf_target = int(selected.get('prefetch_factor', old_pf))

    accu = max(1, int(getattr(__C, 'GRAD_ACCU_STEPS', 1)))
    if bs_target < accu:
        bs_target = accu
    bs_target = (bs_target // accu) * accu
    if bs_target <= 0:
        bs_target = accu

    __C.BATCH_SIZE = int(bs_target)
    __C.NUM_WORKERS = max(0, int(nw_target))
    __C.PREFETCH_FACTOR = max(1, int(pf_target))
    __C.SUB_BATCH_SIZE = int(__C.BATCH_SIZE / accu)
    __C.EVAL_BATCH_SIZE = max(1, int(__C.SUB_BATCH_SIZE / 2))

    print(
        '[AUTO_TUNE] VRAM={:.1f}GB -> BATCH_SIZE {}->{} | NUM_WORKERS {}->{} | PREFETCH_FACTOR {}->{}'.format(
            total_gb,
            old_bs,
            __C.BATCH_SIZE,
            old_nw,
            __C.NUM_WORKERS,
            old_pf,
            __C.PREFETCH_FACTOR
        )
    )

    # Optional model-level overrides from selected rule.
    if 'cmx_grad_checkpoint' in selected:
        old_gc = bool(getattr(__C, 'CMX_GRAD_CHECKPOINT', False))
        __C.CMX_GRAD_CHECKPOINT = bool(selected.get('cmx_grad_checkpoint'))
        print(
            '[AUTO_TUNE] CMX_GRAD_CHECKPOINT {}->{}'.format(
                old_gc,
                bool(__C.CMX_GRAD_CHECKPOINT)
            )
        )


def train_engine(__C, dataset, dataset_eval=None):

    data_size = dataset.data_size
    token_size = dataset.token_size
    ans_size = dataset.ans_size
    pretrained_emb = dataset.pretrained_emb
    _auto_tune_loader_by_vram(__C)

    net = ModelLoader(__C).Net(
        __C,
        pretrained_emb,
        token_size,
        ans_size
    )
    net.cuda()
    net.train()

    if __C.N_GPU > 1:
        net = nn.DataParallel(net, device_ids=__C.DEVICES)

    # Optional: warm-start backbone from a standard MCAN checkpoint (strict=False; lv_gate init)
    if not __C.RESUME:
        _init = getattr(__C, 'PRUNE_INIT_CKPT', '')
        if isinstance(_init, str) and _init.strip() != '':
            from openvqa.models.mcan_lvprune.ckpt_utils import load_checkpoint_into_net
            load_checkpoint_into_net(net, _init.strip())

    # Define Loss Function
    loss_fn = eval('torch.nn.' + __C.LOSS_FUNC_NAME_DICT[__C.LOSS_FUNC] + "(reduction='" + __C.LOSS_REDUCTION + "').cuda()")
    use_amp = bool(getattr(__C, 'USE_AMP', False))
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Load checkpoint if resume training
    if __C.RESUME:
        print(' ========== Resume training')

        if __C.CKPT_PATH is not None:
            print('Warning: Now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = __C.CKPT_PATH
        else:
            path = __C.CKPTS_PATH + \
                   '/ckpt_' + __C.CKPT_VERSION + \
                   '/epoch' + str(__C.CKPT_EPOCH) + '.pkl'

        # Load the network parameters
        print('Loading ckpt from {}'.format(path))
        ckpt = torch.load(path)
        print('Finish!')

        if __C.N_GPU > 1:
            net.load_state_dict(ckpt_proc(ckpt['state_dict']))
        else:
            net.load_state_dict(ckpt['state_dict'])
        start_epoch = ckpt['epoch']

        # Load the optimizer paramters
        optim = get_optim(__C, net, data_size, ckpt['lr_base'])
        optim._step = int(data_size / __C.BATCH_SIZE * start_epoch)
        optim.optimizer.load_state_dict(ckpt['optimizer'])
        
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

    else:
        if ('ckpt_' + __C.VERSION) not in os.listdir(__C.CKPTS_PATH):
            #shutil.rmtree(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)
            os.mkdir(__C.CKPTS_PATH + '/ckpt_' + __C.VERSION)

        optim = get_optim(__C, net, data_size)
        start_epoch = 0

    loss_sum = 0
    named_params = list(net.named_parameters())
    grad_norm = np.zeros(len(named_params))

    # Define multi-thread dataloader
    # if __C.SHUFFLE_MODE in ['external']:
    #     dataloader = Data.DataLoader(
    #         dataset,
    #         batch_size=__C.BATCH_SIZE,
    #         shuffle=False,
    #         num_workers=__C.NUM_WORKERS,
    #         pin_memory=__C.PIN_MEM,
    #         drop_last=True
    #     )
    # else:
    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=__C.BATCH_SIZE,
        shuffle=True,
        num_workers=__C.NUM_WORKERS,
        pin_memory=__C.PIN_MEM,
        drop_last=True
    )
    if __C.NUM_WORKERS > 0:
        dataloader_kwargs['persistent_workers'] = bool(getattr(__C, 'PERSISTENT_WORKERS', True))
        prefetch_factor = int(getattr(__C, 'PREFETCH_FACTOR', 4))
        if prefetch_factor > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
    dataloader = Data.DataLoader(**dataloader_kwargs)

    logfile = open(
        __C.LOG_PATH +
        '/log_run_' + __C.VERSION + '.txt',
        'a+'
    )
    logfile.write(str(__C))
    logfile.close()

    # Training script
    for epoch in range(start_epoch, __C.MAX_EPOCH):
        model_ref = net.module if isinstance(net, nn.DataParallel) else net
        if hasattr(model_ref, 'on_train_epoch_start'):
            model_ref.on_train_epoch_start(epoch)

        # Save log to file
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            '=====================================\nnowTime: ' +
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
            '\n'
        )
        logfile.close()

        # Learning Rate Decay
        if epoch in __C.LR_DECAY_LIST:
            adjust_lr(optim, __C.LR_DECAY_R)

        # Externally shuffle data list
        # if __C.SHUFFLE_MODE == 'external':
        #     dataset.shuffle_list(dataset.ans_list)

        time_start = time.time()
        # Iteration
        for step, (
                frcn_feat_iter,
                grid_feat_iter,
                bbox_feat_iter,
                ques_ix_iter,
                ans_iter
        ) in enumerate(dataloader):

            optim.zero_grad()

            non_blocking = bool(__C.PIN_MEM)
            frcn_feat_iter = frcn_feat_iter.cuda(non_blocking=non_blocking)
            grid_feat_iter = grid_feat_iter.cuda(non_blocking=non_blocking)
            bbox_feat_iter = bbox_feat_iter.cuda(non_blocking=non_blocking)
            ques_ix_iter = ques_ix_iter.cuda(non_blocking=non_blocking)
            ans_iter = ans_iter.cuda(non_blocking=non_blocking)

            loss_tmp = 0
            for accu_step in range(__C.GRAD_ACCU_STEPS):
                loss_tmp = 0

                sub_frcn_feat_iter = \
                    frcn_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_grid_feat_iter = \
                    grid_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_bbox_feat_iter = \
                    bbox_feat_iter[accu_step * __C.SUB_BATCH_SIZE:
                                  (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ques_ix_iter = \
                    ques_ix_iter[accu_step * __C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * __C.SUB_BATCH_SIZE]
                sub_ans_iter = \
                    ans_iter[accu_step * __C.SUB_BATCH_SIZE:
                             (accu_step + 1) * __C.SUB_BATCH_SIZE]

                if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                    amp_ctx = torch.amp.autocast('cuda', enabled=use_amp)
                else:
                    amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                with amp_ctx:
                    pred = net(
                        sub_frcn_feat_iter,
                        sub_grid_feat_iter,
                        sub_bbox_feat_iter,
                        sub_ques_ix_iter
                    )
                    prune_aux = None
                    if isinstance(pred, tuple):
                        pred, prune_aux = pred

                    loss_item = [pred, sub_ans_iter]
                    loss_nonlinear_list = __C.LOSS_FUNC_NONLINEAR[__C.LOSS_FUNC]
                    for item_ix, loss_nonlinear in enumerate(loss_nonlinear_list):
                        if loss_nonlinear in ['flat']:
                            loss_item[item_ix] = loss_item[item_ix].view(-1)
                        elif loss_nonlinear:
                            loss_item[item_ix] = eval('F.' + loss_nonlinear + '(loss_item[item_ix], dim=1)')

                    loss = loss_fn(loss_item[0], loss_item[1])
                    if prune_aux is not None:
                        loss = loss + prune_aux
                    if __C.LOSS_REDUCTION == 'mean':
                        # only mean-reduction needs be divided by grad_accu_steps
                        loss /= __C.GRAD_ACCU_STEPS

                scaler.scale(loss).backward()

                loss_scalar = float(loss.detach().float().item()) * __C.GRAD_ACCU_STEPS
                loss_tmp += loss_scalar
                loss_sum += loss_scalar

            if __C.VERBOSE:
                if dataset_eval is not None:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['val']
                else:
                    mode_str = __C.SPLIT['train'] + '->' + __C.SPLIT['test']

                print("\r[Version %s][Model %s][Dataset %s][Epoch %2d][Step %4d/%4d][%s] Loss: %.4f, Lr: %.2e" % (
                    __C.VERSION,
                    __C.MODEL_USE,
                    __C.DATASET,
                    epoch + 1,
                    step,
                    int(data_size / __C.BATCH_SIZE),
                    mode_str,
                    loss_tmp / __C.SUB_BATCH_SIZE,
                    optim._rate
                ), end='          ', flush=True)

            # Gradient norm clipping
            if __C.GRAD_NORM_CLIP > 0:
                if use_amp:
                    scaler.unscale_(optim.optimizer)
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    __C.GRAD_NORM_CLIP
                )

            # Save the gradient information only when explicitly requested.
            if bool(getattr(__C, 'TRACK_GRAD_NORM', False)):
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * __C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

            if use_amp:
                optim._step += 1
                rate = optim.rate()
                for p in optim.optimizer.param_groups:
                    lr_scale = p['lr_scale'] if 'lr_scale' in p else 1.0
                    p['lr'] = rate * lr_scale
                optim._rate = rate
                scaler.step(optim.optimizer)
                scaler.update()
            else:
                optim.step()

        time_end = time.time()
        elapse_time = time_end-time_start
        print('Finished in {}s'.format(int(elapse_time)))
        epoch_finish = epoch + 1

        # Save checkpoint
        if __C.N_GPU > 1:
            state = {
                'state_dict': net.module.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        else:
            state = {
                'state_dict': net.state_dict(),
                'optimizer': optim.optimizer.state_dict(),
                'lr_base': optim.lr_base,
                'epoch': epoch_finish
            }
        torch.save(
            state,
            __C.CKPTS_PATH +
            '/ckpt_' + __C.VERSION +
            '/epoch' + str(epoch_finish) +
            '.pkl'
        )

        # Logging
        logfile = open(
            __C.LOG_PATH +
            '/log_run_' + __C.VERSION + '.txt',
            'a+'
        )
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / data_size) +
            ', Lr: ' + str(optim._rate) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) + 
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        logfile.close()

        # Eval after every epoch
        if dataset_eval is not None:
            test_engine(
                __C,
                dataset_eval,
                state_dict=net.state_dict(),
                validation=True
            )

        # if self.__C.VERBOSE:
        #     logfile = open(
        #         self.__C.LOG_PATH +
        #         '/log_run_' + self.__C.VERSION + '.txt',
        #         'a+'
        #     )
        #     for name in range(len(named_params)):
        #         logfile.write(
        #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
        #                 str(name),
        #                 named_params[name][0],
        #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
        #             )
        #         )
        #     logfile.write('\n')
        #     logfile.close()

        loss_sum = 0
        grad_norm = np.zeros(len(named_params))
