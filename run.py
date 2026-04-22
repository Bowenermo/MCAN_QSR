# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.models.model_loader import CfgLoader
from utils.exec import Execution
import argparse, yaml


def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='OpenVQA Args')

    parser.add_argument('--RUN', dest='RUN_MODE',
                      choices=['train', 'val', 'test'],
                      help='{train, val, test}',
                      type=str, required=True)

    parser.add_argument('--MODEL', dest='MODEL',
                      choices=[
                           'mcan_small',
                           'mcan_large',
                           'mcan_small_patch',
                           'mcan_large_patch',
                           'mcan_lvprune_small',
                           'mcan_lvprune_large',
                           'mcan_lvprune_small_patch',
                           'mcan_lvprune_large_patch',
                           'mcan_convmixer_e2e_small',
                           'mcan_convmixer_e2e_small_fast',
                           'mcan_convmixer_p12_small',
                           'mcan_convmixer_p123_small',
                           'mcan_cmx_gridppm_small',
                           'mcan_cmx_gridppm_small_fastv2',
                           'mcan_qsr_small',
                           'mcan_qsr_small_fastv2_auto',
                           'mcan_qsr_noq_small',
                           'mcan_qsr_noq_small_fastv2_auto',
                           'ban_4',
                           'ban_8',
                           'mfb',
                           'mfh',
                           'mem',
                           'butd',
                           'mmnasnet'
                           ]
                        ,
                      help='{'
                           'mcan_small,'
                           'mcan_large,'
                           'mcan_small_patch,'
                           'mcan_large_patch,'
                           'mcan_convmixer_e2e_small,'
                           'mcan_convmixer_e2e_small_fast,'
                           'mcan_convmixer_p12_small,'
                           'mcan_convmixer_p123_small,'
                           'mcan_cmx_gridppm_small,'
                           'mcan_cmx_gridppm_small_fastv2,'
                           'mcan_qsr_small,'
                           'mcan_qsr_small_fastv2_auto,'
                           'mcan_qsr_noq_small,'
                           'mcan_qsr_noq_small_fastv2_auto,'
                           'ban_4,'
                           'ban_8,'
                           'mfb,'
                           'mfh,'
                           'butd,'
                           'mmnasnet,'
                           '}'
                        ,
                      type=str, required=True)

    parser.add_argument('--DATASET', dest='DATASET',
                      choices=['vqa', 'gqa', 'clevr'],
                      help='{'
                           'vqa,'
                           'gqa,'
                           'clevr,'
                           '}'
                        ,
                      type=str, required=True)

    parser.add_argument('--SPLIT', dest='TRAIN_SPLIT',
                      choices=['train', 'train+val', 'train+val+vg'],
                      help="set training split, "
                           "vqa: {'train', 'train+val', 'train+val+vg'}"
                           "gqa: {'train', 'train+val'}"
                           "clevr: {'train', 'train+val'}"
                        ,
                      type=str)

    parser.add_argument('--EVAL_EE', dest='EVAL_EVERY_EPOCH',
                      choices=['True', 'False'],
                      help='True: evaluate the val split when an epoch finished,'
                           'False: do not evaluate on local',
                      type=str)

    parser.add_argument('--SAVE_PRED', dest='TEST_SAVE_PRED',
                      choices=['True', 'False'],
                      help='True: save the prediction vectors,'
                           'False: do not save the prediction vectors',
                      type=str)

    parser.add_argument('--BS', dest='BATCH_SIZE',
                      help='batch size in training',
                      type=int)

    parser.add_argument('--GPU', dest='GPU',
                      help="gpu choose, eg.'0, 1, 2, ...'",
                      type=str)

    parser.add_argument('--SEED', dest='SEED',
                      help='fix random seed',
                      type=int)

    parser.add_argument('--VERSION', dest='VERSION',
                      help='version control',
                      type=str)

    parser.add_argument('--RESUME', dest='RESUME',
                      choices=['True', 'False'],
                      help='True: use checkpoint to resume training,'
                           'False: start training with random init',
                      type=str)

    parser.add_argument('--RESUME_WEIGHTS_ONLY', dest='RESUME_WEIGHTS_ONLY',
                      choices=['True', 'False'],
                      help='True: load model weights from checkpoint but reset optimizer/lr schedule',
                      type=str)

    parser.add_argument('--CKPT_V', dest='CKPT_VERSION',
                      help='checkpoint version',
                      type=str)

    parser.add_argument('--CKPT_E', dest='CKPT_EPOCH',
                      help='checkpoint epoch',
                      type=int)

    parser.add_argument('--CKPT_PATH', dest='CKPT_PATH',
                      help='load checkpoint path, we '
                           'recommend that you use '
                           'CKPT_VERSION and CKPT_EPOCH '
                           'instead, it will override'
                           'CKPT_VERSION and CKPT_EPOCH',
                      type=str)

    parser.add_argument('--ACCU', dest='GRAD_ACCU_STEPS',
                      help='split batch to reduce gpu memory usage',
                      type=int)

    parser.add_argument('--NW', dest='NUM_WORKERS',
                      help='multithreaded loading to accelerate IO',
                      type=int)

    parser.add_argument('--PINM', dest='PIN_MEM',
                      choices=['True', 'False'],
                      help='True: use pin memory, False: not use pin memory',
                      type=str)

    parser.add_argument('--PERSISTW', dest='PERSISTENT_WORKERS',
                      choices=['True', 'False'],
                      help='True: keep dataloader workers alive across epochs',
                      type=str)

    parser.add_argument('--PREFETCH', dest='PREFETCH_FACTOR',
                      help='dataloader prefetch factor (num_workers > 0)',
                      type=int)

    parser.add_argument('--VERB', dest='VERBOSE',
                      choices=['True', 'False'],
                      help='True: verbose print, False: simple print',
                      type=str)

    parser.add_argument('--SKIP_BAD_FEAT', dest='SKIP_BAD_FEAT',
                      choices=['True', 'False'],
                      help='True: skip broken feature files at runtime, False: raise error',
                      type=str)

    parser.add_argument('--VQA_FEAT_ROOT', dest='VQA_FEAT_ROOT',
                      help='override VQA visual feature root path; '
                           'expects train2014/ val2014/ test2015 subfolders',
                      type=str)

    parser.add_argument('--PRUNE_INIT_CKPT', dest='PRUNE_INIT_CKPT',
                      help='(mcan_lvprune) load MCAN checkpoint into backbone with strict=False before train',
                      type=str)

    parser.add_argument('--FREEZE', dest='FREEZE_BACKBONE',
                      choices=['True', 'False'],
                      help='(mcan_lvprune) True: train only lv_gate modules',
                      type=str)

    parser.add_argument('--AMP', dest='USE_AMP',
                      choices=['True', 'False'],
                      help='True: enable mixed precision training (AMP)',
                      type=str)

    parser.add_argument('--CUDNN_BENCHMARK', dest='CUDNN_BENCHMARK',
                      choices=['True', 'False'],
                      help='True: enable cuDNN benchmark for speed on fixed input shapes',
                      type=str)

    parser.add_argument('--CUDNN_DETERMINISTIC', dest='CUDNN_DETERMINISTIC',
                      choices=['True', 'False'],
                      help='True: deterministic cuDNN kernels (slower but reproducible)',
                      type=str)

    parser.add_argument('--TRACK_GRAD_NORM', dest='TRACK_GRAD_NORM',
                      choices=['True', 'False'],
                      help='True: compute per-parameter grad norm each step (slower)',
                      type=str)

    parser.add_argument('--BAD_LOG_INT', dest='BAD_SAMPLE_LOG_INTERVAL',
                      help='print broken-sample logs every N skips after first occurrence',
                      type=int)

    parser.add_argument('--SUBSET_RATIO', dest='DATA_SUBSET_RATIO',
                      help='use a subset ratio of current split for quick runs, e.g. 0.01 for 1%',
                      type=float)


    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    cfg_file = "configs/{}/{}.yml".format(args.DATASET, args.MODEL)
    with open(cfg_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)

    __C = CfgLoader(yaml_dict['MODEL_USE']).load()
    args = __C.str_to_bool(args)
    args_dict = __C.parse_to_dict(args)

    args_dict = {**yaml_dict, **args_dict}
    __C.add_args(args_dict)
    __C.proc()

    print('Hyper Parameters:')
    print(__C)

    execution = Execution(__C)
    execution.run(__C.RUN_MODE)




