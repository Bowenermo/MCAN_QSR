# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from openvqa.core.path_cfgs import PATH
import os, torch, random
import numpy as np
from types import MethodType


class BaseCfgs(PATH):
    def __init__(self):
        super(BaseCfgs, self).__init__()

        # Set Devices
        # If use multi-gpu training, you can set e.g.'0, 1, 2' instead
        self.GPU = '0'

        # Set Seed For CPU And GPUs
        self.SEED = random.randint(0, 9999999)

        # -------------------------
        # ---- Version Control ----
        # -------------------------

        # You can set a name to start new training
        self.VERSION = str(self.SEED)

        # Use checkpoint to resume training
        self.RESUME = False
        # If True with RESUME, only load model weights and reinitialize optimizer/lr schedule.
        self.RESUME_WEIGHTS_ONLY = False

        # Resume training version or testing version
        self.CKPT_VERSION = self.VERSION

        # Resume training epoch or testing epoch
        self.CKPT_EPOCH = 0

        # if set 'CKPT_PATH', -> 'CKPT_VERSION' and 'CKPT_EPOCH' will not work any more
        self.CKPT_PATH = None

        # Print loss every iteration
        self.VERBOSE = True

        # If True, skip broken visual feature files during data loading.
        # If False, training stops at first broken file.
        self.SKIP_BAD_FEAT = False

        # Max retries when skipping broken features in one sample fetch.
        self.MAX_BAD_FEAT_RETRY = 20


        # ------------------------------
        # ---- Data Provider Params ----
        # ------------------------------

        self.MODEL = ''

        self.MODEL_USE = ''

        self.DATASET = ''

        # Run as 'train' 'val' or 'test'
        self.RUN_MODE = ''

        # Set True to evaluate offline when an epoch finished
        # (only work when train with 'train' split)
        self.EVAL_EVERY_EPOCH = True

        # Set True to save the prediction vector
        # (use in ensemble)
        self.TEST_SAVE_PRED = False


        # A external method to set train split
        # will override the SPLIT['train']
        self.TRAIN_SPLIT = 'train'

        # Optional override for VQA feature root.
        # Expected folder structure:
        #   <root>/train2014/*.npz
        #   <root>/val2014/*.npz
        #   <root>/test2015/*.npz
        # Keep empty to use the default region-based features.
        self.VQA_FEAT_ROOT = ''

        # If True, VQA loader reads raw images directly instead of .npz features.
        self.USE_RAW_IMAGE_INPUT = False

        # Set True to use pretrained GloVe word embedding
        # (GloVe: spaCy https://spacy.io/)
        self.USE_GLOVE = True

        # Word embedding matrix size
        # (token size x WORD_EMBED_SIZE)
        self.WORD_EMBED_SIZE = 300

        # All features size
        self.FEAT_SIZE = {
            'vqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'gqa': {
                'FRCN_FEAT_SIZE': (100, 2048),
                'GRID_FEAT_SIZE': (49, 2048),
                'BBOX_FEAT_SIZE': (100, 5),
            },
            'clevr': {
                'GRID_FEAT_SIZE': (196, 1024),
            },
        }

        # Set if bbox_feat need be normalize by image size, default: False
        self.BBOX_NORMALIZE = False

        # Default training batch size: 64
        self.BATCH_SIZE = 64

        # Multi-thread I/O
        self.NUM_WORKERS = 8

        # Use pin memory
        # (Warning: pin memory can accelerate GPU loading but may
        # increase the CPU memory usage when NUM_WORKS is big)
        self.PIN_MEM = True
        self.PERSISTENT_WORKERS = True
        self.PREFETCH_FACTOR = 4

        # Large model can not training with batch size 64
        # Gradient accumulate can split batch to reduce gpu memory usage
        # (Warning: BATCH_SIZE should be divided by GRAD_ACCU_STEPS)
        self.GRAD_ACCU_STEPS = 1

        # mcan_lvprune: optional warm-start from MCAN .pkl (see train_engine)
        self.PRUNE_INIT_CKPT = ''
        # mcan_lvprune: if True, only train submodules whose name contains PRUNE_TRAIN_NAME_SUBSTR (set in mcan_lvprune cfg)
        self.FREEZE_BACKBONE = False

        # If True, enable mixed precision training (AMP) in train_engine.
        self.USE_AMP = False

        # cuDNN speed/reproducibility knobs.
        # Keep deterministic by default; set BENCHMARK=True and DETERMINISTIC=False
        # for maximum throughput in fixed-shape training.
        self.CUDNN_BENCHMARK = False
        self.CUDNN_DETERMINISTIC = True

        # If True, compute and accumulate grad norm for every parameter each step.
        # This is expensive and disabled by default for speed.
        self.TRACK_GRAD_NORM = False

        # Print detailed broken-sample logs at this interval after first occurrence.
        self.BAD_SAMPLE_LOG_INTERVAL = 100

        # Use a fraction of dataset samples for quick experiments/debugging.
        # 1.0 means full data; 0.01 means 1% data.
        self.DATA_SUBSET_RATIO = 1.0


        # --------------------------
        # ---- Optimizer Params ----
        # --------------------------

        # Define the loss function
        '''
        Loss(case-sensitive): 
        'ce'    : Cross Entropy -> NLLLoss(LogSoftmax(output), label) = CrossEntropyLoss(output, label)
        'bce'   : Binary Cross Entropy -> BCELoss(Sigmoid(output), label) = BCEWithLogitsLoss(output, label)
        'kld'   : Kullback-Leibler Divergence -> KLDivLoss(LogSoftmax(output), Softmax(label))
        'mse'   : Mean Squared Error -> MSELoss(output, label)
        
        Reduction(case-sensitive):
        'none': no reduction will be applied
        'elementwise_mean': the sum of the output will be divided by the number of elements in the output
        'sum': the output will be summed
        '''
        self.LOSS_FUNC = ''
        self.LOSS_REDUCTION = ''


        # The base learning rate
        self.LR_BASE = 0.0001

        # Learning rate decay ratio
        self.LR_DECAY_R = 0.2

        # Learning rate decay at {x, y, z...} epoch
        self.LR_DECAY_LIST = [10, 12]

        # Warmup epoch lr*{1/(n+1), 2/(n+1), ... , n/(n+1)}
        self.WARMUP_EPOCH = 3

        # Max training epoch
        self.MAX_EPOCH = 13

        # Gradient clip
        # (default: -1 means not using)
        self.GRAD_NORM_CLIP = -1

        # Optimizer
        '''
        Optimizer(case-sensitive): 
        'Adam'      : default -> {betas:(0.9, 0.999), eps:1e-8, weight_decay:0, amsgrad:False}
        'Adamax'    : default -> {betas:(0.9, 0.999), eps:1e-8, weight_decay:0}
        'RMSprop'   : default -> {alpha:0.99, eps:1e-8, weight_decay:0, momentum:0, centered:False}
        'SGD'       : default -> {momentum:0, dampening:0, weight_decay:0, nesterov:False}
        'Adadelta'  : default -> {rho:0.9, eps:1e-6, weight_decay:0}
        'Adagrad'   : default -> {lr_decay:0, weight_decay:0, initial_accumulator_value:0}
        
        In YML files:
        If you want to self-define the optimizer parameters, set a dict named OPT_PARAMS contains the keys you want to modify.
         !!! Warning: keys: ['params, 'lr'] should not be set. 
         !!! Warning: To avoid ambiguity, the value of keys should be defined as string type.
        If you not define the OPT_PARAMS, all parameters of optimizer will be set as default.
        Example:
        mcan_small.yml ->
            OPT: Adam
            OPT_PARAMS: {betas: '(0.9, 0.98)', eps: '1e-9'}
        '''
        # case-sensitive
        self.OPT = ''
        self.OPT_PARAMS = {}


    def str_to_bool(self, args):
        bool_list = [
            'EVAL_EVERY_EPOCH',
            'TEST_SAVE_PRED',
            'RESUME',
            'RESUME_WEIGHTS_ONLY',
            'PIN_MEM',
            'VERBOSE',
            'SKIP_BAD_FEAT',
            'FREEZE_BACKBONE',
            'USE_AMP',
            'PERSISTENT_WORKERS',
            'CUDNN_BENCHMARK',
            'CUDNN_DETERMINISTIC',
            'TRACK_GRAD_NORM',
        ]

        for arg in dir(args):
            if arg in bool_list and getattr(args, arg) is not None:
                setattr(args, arg, eval(getattr(args, arg)))

        return args


    def parse_to_dict(self, args):
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith('_') and not isinstance(getattr(args, arg), MethodType):
                if getattr(args, arg) is not None:
                    args_dict[arg] = getattr(args, arg)

        return args_dict


    def add_args(self, args_dict):
        for arg in args_dict:
            setattr(self, arg, args_dict[arg])


    def proc(self):
        assert self.RUN_MODE in ['train', 'val', 'test']
        assert 0 < float(getattr(self, 'DATA_SUBSET_RATIO', 1.0)) <= 1.0

        # ------------ Devices setup
        os.environ['CUDA_VISIBLE_DEVICES'] = self.GPU
        self.N_GPU = len(self.GPU.split(','))
        self.DEVICES = [_ for _ in range(self.N_GPU)]
        torch.set_num_threads(2)

        # ------------ Optional feature-path override / raw-image mode
        if self.DATASET == 'vqa':
            if bool(getattr(self, 'USE_RAW_IMAGE_INPUT', False)):
                self.FEATS_PATH['vqa']['train'] = self.DATA_PATH['vqa'] + '/raw/train2014'
                self.FEATS_PATH['vqa']['val'] = self.DATA_PATH['vqa'] + '/raw/val2014'
                self.FEATS_PATH['vqa']['test'] = self.DATA_PATH['vqa'] + '/raw/test2015'
            elif isinstance(self.VQA_FEAT_ROOT, str) and self.VQA_FEAT_ROOT.strip() != '':
                vqa_feat_root = self.VQA_FEAT_ROOT.rstrip('/')
                self.FEATS_PATH['vqa']['train'] = vqa_feat_root + '/train2014'
                self.FEATS_PATH['vqa']['val'] = vqa_feat_root + '/val2014'
                self.FEATS_PATH['vqa']['test'] = vqa_feat_root + '/test2015'


        # ------------ Path check
        self.check_path(self.DATASET)


        # ------------ Model setup (Deprecated)
        # self.MODEL_USE = self.MODEL.split('_')[0]


        # ------------ Seed setup
        # fix pytorch seed
        torch.manual_seed(self.SEED)
        if self.N_GPU < 2:
            torch.cuda.manual_seed(self.SEED)
        else:
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = bool(getattr(self, 'CUDNN_DETERMINISTIC', True))
        torch.backends.cudnn.benchmark = bool(getattr(self, 'CUDNN_BENCHMARK', False)) and \
            (not torch.backends.cudnn.deterministic)

        # fix numpy seed
        np.random.seed(self.SEED)

        # fix random seed
        random.seed(self.SEED)

        if self.CKPT_PATH is not None:
            print("Warning: you are now using 'CKPT_PATH' args, "
                  "'CKPT_VERSION' and 'CKPT_EPOCH' will not work")
            self.CKPT_VERSION = self.CKPT_PATH.split('/')[-1] + '_' + str(random.randint(0, 9999999))


        # ------------ Split setup
        self.SPLIT = self.SPLITS[self.DATASET]
        self.SPLIT['train'] = self.TRAIN_SPLIT
        if self.SPLIT['val'] in self.SPLIT['train'].split('+') or self.RUN_MODE not in ['train']:
            self.EVAL_EVERY_EPOCH = False

        if self.RUN_MODE not in ['test']:
            self.TEST_SAVE_PRED = False


        # ------------ Gradient accumulate setup
        assert self.BATCH_SIZE % self.GRAD_ACCU_STEPS == 0
        self.SUB_BATCH_SIZE = int(self.BATCH_SIZE / self.GRAD_ACCU_STEPS)

        # Set small eval batch size will reduce gpu memory usage
        self.EVAL_BATCH_SIZE = int(self.SUB_BATCH_SIZE / 2)


        # ------------ Loss process
        assert self.LOSS_FUNC in ['ce', 'bce', 'kld', 'mse']
        assert self.LOSS_REDUCTION in ['none', 'elementwise_mean', 'sum']

        self.LOSS_FUNC_NAME_DICT = {
            'ce': 'CrossEntropyLoss',
            'bce': 'BCEWithLogitsLoss',
            'kld': 'KLDivLoss',
            'mse': 'MSELoss',
        }

        self.LOSS_FUNC_NONLINEAR = {
            'ce': [None, 'flat'],
            'bce': [None, None],
            'kld': ['log_softmax', None],
            'mse': [None, None],
        }

        self.TASK_LOSS_CHECK = {
            'vqa': ['bce', 'kld'],
            'gqa': ['ce'],
            'clevr': ['ce'],
        }

        assert self.LOSS_FUNC in self.TASK_LOSS_CHECK[self.DATASET], \
            self.DATASET + 'task only support' + str(self.TASK_LOSS_CHECK[self.DATASET]) + 'loss.' + \
            'Modify the LOSS_FUNC in configs to get a better score.'


        # ------------ Optimizer parameters process
        assert self.OPT in ['Adam', 'Adamax', 'RMSprop', 'SGD', 'Adadelta', 'Adagrad']
        optim = getattr(torch.optim, self.OPT)
        default_params_dict = dict(zip(optim.__init__.__code__.co_varnames[3: optim.__init__.__code__.co_argcount],
                                       optim.__init__.__defaults__[1:]))

        def all(iterable):
            for element in iterable:
                if not element:
                    return False
            return True
        assert all(list(map(lambda x: x in default_params_dict, self.OPT_PARAMS)))

        for key in self.OPT_PARAMS:
            if isinstance(self.OPT_PARAMS[key], str):
                self.OPT_PARAMS[key] = eval(self.OPT_PARAMS[key])
            else:
                print("To avoid ambiguity, set the value of 'OPT_PARAMS' to string type")
                exit(-1)
        self.OPT_PARAMS = {**default_params_dict, **self.OPT_PARAMS}

    def __str__(self):
        __C_str = ''
        for attr in dir(self):
            if not attr.startswith('__') and not isinstance(getattr(self, attr), MethodType):
                __C_str += '{ %-17s }->' % attr + str(getattr(self, attr)) + '\n'

        return __C_str


#
#
# if __name__ == '__main__':
#     __C = Cfgs()
#     __C.proc()





