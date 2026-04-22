# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import numpy as np
import glob, json, re, en_vectors_web_lg
import torch
import os
from openvqa.core.base_dataset import BaseDataSet
from openvqa.utils.ans_punct import prep_ans

class DataSet(BaseDataSet):
    def __init__(self, __C):
        super(DataSet, self).__init__()
        self.__C = __C
        self.use_raw_image_input = bool(getattr(__C, 'USE_RAW_IMAGE_INPUT', False))
        self.data_subset_ratio = float(getattr(__C, 'DATA_SUBSET_RATIO', 1.0))

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        frcn_feat_path_list = []
        if not self.use_raw_image_input:
            frcn_feat_path_list = \
                glob.glob(__C.FEATS_PATH[__C.DATASET]['train'] + '/*.npz') + \
                glob.glob(__C.FEATS_PATH[__C.DATASET]['val'] + '/*.npz') + \
                glob.glob(__C.FEATS_PATH[__C.DATASET]['test'] + '/*.npz')

        # Loading question word list
        stat_ques_list = \
            json.load(open(__C.RAW_PATH[__C.DATASET]['train'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['val'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['test'], 'r'))['questions'] + \
            json.load(open(__C.RAW_PATH[__C.DATASET]['vg'], 'r'))['questions']

        # Loading answer word list
        # stat_ans_list = \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['train-anno'], 'r'))['annotations'] + \
        #     json.load(open(__C.RAW_PATH[__C.DATASET]['val-anno'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.RAW_PATH[__C.DATASET][split], 'r'))['questions']
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.RAW_PATH[__C.DATASET][split + '-anno'], 'r'))['annotations']

        self._apply_subset_ratio()

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print(' ========== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        if self.use_raw_image_input:
            from torchvision import transforms
            self.raw_image_transform = transforms.Compose([
                transforms.Resize((__C.RAW_IMAGE_INPUT_SIZE, __C.RAW_IMAGE_INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=tuple(__C.RAW_IMAGE_MEAN),
                    std=tuple(__C.RAW_IMAGE_STD)
                ),
            ])

            raw_img_path_list = \
                glob.glob(__C.DATA_PATH['vqa'] + '/raw/train2014/*.jpg') + \
                glob.glob(__C.DATA_PATH['vqa'] + '/raw/val2014/*.jpg') + \
                glob.glob(__C.DATA_PATH['vqa'] + '/raw/test2015/*.jpg')
            self.iid_to_img_path = self.img_feat_path_load(raw_img_path_list)
        else:
            # {image id} -> {image feature absolutely path}
            self.iid_to_frcn_feat_path = self.img_feat_path_load(frcn_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = self.ques_load(self.ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = self.tokenize(stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print(' ========== Question token vocab size:', self.token_size)

        # Answers statistic
        self.ans_to_ix, self.ix_to_ans = self.ans_stat('openvqa/datasets/vqa/answer_dict.json')
        # self.ans_to_ix, self.ix_to_ans = self.ans_stat(stat_ans_list, ans_freq=8)
        self.ans_size = self.ans_to_ix.__len__()
        print(' ========== Answer token vocab size (occur more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')
        self.skip_sample_count = 0
        self.skip_raw_image_count = 0
        self.skip_feat_file_count = 0
        self._reported_broken_iids = set()

    def _apply_subset_ratio(self):
        # Keep validation/test full-size by default.
        if self.__C.RUN_MODE not in ['train']:
            return

        if self.data_subset_ratio >= 1.0:
            return

        total = len(self.ans_list)
        keep = max(1, int(total * self.data_subset_ratio))
        rng = np.random.RandomState(int(self.__C.SEED))
        select_idx = np.sort(rng.choice(total, size=keep, replace=False))
        self.ans_list = [self.ans_list[i] for i in select_idx]
        print(' ========== DATA_SUBSET_RATIO(train): {:.4f} -> {} / {}'.format(
            self.data_subset_ratio, keep, total
        ))


    def __getitem__(self, idx):
        max_retry = int(getattr(self.__C, 'MAX_BAD_FEAT_RETRY', 20))
        # In raw-image mode, keep training robust by default.
        skip_bad_feat = bool(getattr(self.__C, 'SKIP_BAD_FEAT', False)) or self.use_raw_image_input

        cur_idx = idx
        for retry in range(max_retry + 1):
            iid = 'unknown'
            try:
                ques_ix_iter, ans_iter, iid = self.load_ques_ans(cur_idx)
                frcn_feat_iter, grid_feat_iter, bbox_feat_iter = self.load_img_feats(cur_idx, iid)

                return \
                    torch.from_numpy(frcn_feat_iter), \
                    torch.from_numpy(grid_feat_iter), \
                    torch.from_numpy(bbox_feat_iter), \
                    torch.from_numpy(ques_ix_iter), \
                    torch.from_numpy(ans_iter)
            except Exception as err:
                if not skip_bad_feat:
                    raise

                err_msg = repr(err)
                self.skip_sample_count += 1
                if 'raw image' in err_msg:
                    self.skip_raw_image_count += 1
                else:
                    self.skip_feat_file_count += 1

                worker_info = torch.utils.data.get_worker_info()
                worker_id = worker_info.id if worker_info is not None else 0
                pid = os.getpid()

                log_interval = int(getattr(self.__C, 'BAD_SAMPLE_LOG_INTERVAL', 100))
                iid_key = iid
                first_seen = iid_key not in self._reported_broken_iids
                if first_seen:
                    self._reported_broken_iids.add(iid_key)
                if first_seen or (log_interval > 0 and self.skip_sample_count % log_interval == 0):
                    print(
                        "[WARN][VQA DataSet] skip broken sample idx={} retry={}/{} "
                        "| worker={} pid={} | skipped_total={} raw_image_skipped={} feat_skipped={} "
                        "| unique_broken_iid={} | iid={} | err={}".format(
                            cur_idx,
                            retry + 1,
                            max_retry + 1,
                            worker_id,
                            pid,
                            self.skip_sample_count,
                            self.skip_raw_image_count,
                            self.skip_feat_file_count,
                            len(self._reported_broken_iids),
                            iid_key,
                            err_msg
                        ),
                        flush=True
                    )
                cur_idx = (cur_idx + 1) % self.data_size

        raise RuntimeError(
            "Exceeded MAX_BAD_FEAT_RETRY={} while fetching data. "
            "Set SKIP_BAD_FEAT=False to fail-fast and inspect exact traceback.".format(max_retry)
        )



    def img_feat_path_load(self, path_list):
        iid_to_path = {}

        for ix, path in enumerate(path_list):
            iid = str(int(path.split('/')[-1].split('_')[-1].split('.')[0]))
            # print(iid)
            iid_to_path[iid] = path

        return iid_to_path


    def ques_load(self, ques_list):
        qid_to_ques = {}

        for ques in ques_list:
            qid = str(ques['question_id'])
            qid_to_ques[qid] = ques

        return qid_to_ques


    def tokenize(self, stat_ques_list, use_glove):
        token_to_ix = {
            'PAD': 0,
            'UNK': 1,
            'CLS': 2,
        }

        spacy_tool = None
        pretrained_emb = []
        if use_glove:
            spacy_tool = en_vectors_web_lg.load()
            pretrained_emb.append(spacy_tool('PAD').vector)
            pretrained_emb.append(spacy_tool('UNK').vector)
            pretrained_emb.append(spacy_tool('CLS').vector)

        for ques in stat_ques_list:
            words = re.sub(
                r"([.,'!?\"()*#:;])",
                '',
                ques['question'].lower()
            ).replace('-', ' ').replace('/', ' ').split()

            for word in words:
                if word not in token_to_ix:
                    token_to_ix[word] = len(token_to_ix)
                    if use_glove:
                        pretrained_emb.append(spacy_tool(word).vector)

        pretrained_emb = np.array(pretrained_emb)

        return token_to_ix, pretrained_emb


    # def ans_stat(self, stat_ans_list, ans_freq):
    #     ans_to_ix = {}
    #     ix_to_ans = {}
    #     ans_freq_dict = {}
    #
    #     for ans in stat_ans_list:
    #         ans_proc = prep_ans(ans['multiple_choice_answer'])
    #         if ans_proc not in ans_freq_dict:
    #             ans_freq_dict[ans_proc] = 1
    #         else:
    #             ans_freq_dict[ans_proc] += 1
    #
    #     ans_freq_filter = ans_freq_dict.copy()
    #     for ans in ans_freq_dict:
    #         if ans_freq_dict[ans] <= ans_freq:
    #             ans_freq_filter.pop(ans)
    #
    #     for ans in ans_freq_filter:
    #         ix_to_ans[ans_to_ix.__len__()] = ans
    #         ans_to_ix[ans] = ans_to_ix.__len__()
    #
    #     return ans_to_ix, ix_to_ans

    def ans_stat(self, json_file):
        ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))

        return ans_to_ix, ix_to_ans



    # ----------------------------------------------
    # ---- Real-Time Processing Implementations ----
    # ----------------------------------------------

    def load_ques_ans(self, idx):
        if self.__C.RUN_MODE in ['train']:
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]
            iid = str(ans['image_id'])

            # Process question
            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            # Process answer
            ans_iter = self.proc_ans(ans, self.ans_to_ix)

            return ques_ix_iter, ans_iter, iid

        else:
            ques = self.ques_list[idx]
            iid = str(ques['image_id'])

            ques_ix_iter = self.proc_ques(ques, self.token_to_ix, max_token=14)

            return ques_ix_iter, np.zeros(1), iid


    def load_img_feats(self, idx, iid):
        if self.use_raw_image_input:
            from PIL import Image
            if iid not in self.iid_to_img_path:
                raise RuntimeError("Raw image file not found for iid={}".format(iid))

            img_path = self.iid_to_img_path[iid]
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    img_tensor = self.raw_image_transform(img)
            except Exception as err:
                raise RuntimeError(
                    "Failed to load raw image: {} | iid={} | err={}".format(img_path, iid, repr(err))
                )

            # Keep interface unchanged for training engine.
            frcn_feat_iter = img_tensor.numpy().astype(np.float32)
            grid_feat_iter = np.zeros(1, dtype=np.float32)
            bbox_feat_iter = np.zeros((1, 4), dtype=np.float32)
            return frcn_feat_iter, grid_feat_iter, bbox_feat_iter

        feat_path = self.iid_to_frcn_feat_path[iid]
        try:
            frcn_feat = np.load(feat_path)
        except Exception as err:
            raise RuntimeError(
                "Failed to load npz feature file: {} | iid={} | err={}".format(feat_path, iid, repr(err))
            )

        if 'x' not in frcn_feat.files:
            raise RuntimeError(
                "Feature file missing key 'x': {} | iid={} | available_keys={}".format(
                    feat_path, iid, frcn_feat.files
                )
            )
        if 'bbox' not in frcn_feat.files:
            raise RuntimeError(
                "Feature file missing key 'bbox': {} | iid={} | available_keys={}".format(
                    feat_path, iid, frcn_feat.files
                )
            )
        if 'image_h' not in frcn_feat.files or 'image_w' not in frcn_feat.files:
            raise RuntimeError(
                "Feature file missing image size keys ('image_h'/'image_w'): {} | iid={} | available_keys={}".format(
                    feat_path, iid, frcn_feat.files
                )
            )

        frcn_feat_x = frcn_feat['x'].transpose((1, 0))
        frcn_feat_iter = self.proc_img_feat(frcn_feat_x, img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['FRCN_FEAT_SIZE'][0])

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['image_h'], frcn_feat['image_w'])
            ),
            img_feat_pad_size=self.__C.FEAT_SIZE['vqa']['BBOX_FEAT_SIZE'][0]
        )
        grid_feat_iter = np.zeros(1)

        return frcn_feat_iter, grid_feat_iter, bbox_feat_iter



    # ------------------------------------
    # ---- Real-Time Processing Utils ----
    # ------------------------------------

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat


    def proc_bbox_feat(self, bbox, img_shape):
        if self.__C.BBOX_NORMALIZE:
            bbox_nm = np.zeros((bbox.shape[0], 4), dtype=np.float32)

            bbox_nm[:, 0] = bbox[:, 0] / float(img_shape[1])
            bbox_nm[:, 1] = bbox[:, 1] / float(img_shape[0])
            bbox_nm[:, 2] = bbox[:, 2] / float(img_shape[1])
            bbox_nm[:, 3] = bbox[:, 3] / float(img_shape[0])
            return bbox_nm
        # bbox_feat[:, 4] = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1]) / float(img_shape[0] * img_shape[1])

        return bbox


    def proc_ques(self, ques, token_to_ix, max_token):
        ques_ix = np.zeros(max_token, np.int64)

        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in token_to_ix:
                ques_ix[ix] = token_to_ix[word]
            else:
                ques_ix[ix] = token_to_ix['UNK']

            if ix + 1 == max_token:
                break

        return ques_ix


    def get_score(self, occur):
        if occur == 0:
            return .0
        elif occur == 1:
            return .3
        elif occur == 2:
            return .6
        elif occur == 3:
            return .9
        else:
            return 1.


    def proc_ans(self, ans, ans_to_ix):
        ans_score = np.zeros(ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        if self.__C.LOSS_FUNC in ['kld']:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = ans_prob_dict[ans_] / 10.
        else:
            for ans_ in ans_prob_dict:
                if ans_ in ans_to_ix:
                    ans_score[ans_to_ix[ans_]] = self.get_score(ans_prob_dict[ans_])

        return ans_score
