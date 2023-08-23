# encoding: utf-8
import os

import torch
import torch.distributed as dist

from yolox.data import get_yolox_datadir
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 32  # model의 클래스 개수(배경 포함 x)
        self.depth = 0.33
        self.width = 0.50
        self.max_epoch = 60  # 학습에 사용할 에폭 개수
        self.data_num_workers = 0
        self.input_size = (640, 640) # input size 설정

        # YOLOX-ROBOTIS/dataset/voc 내에 있는 데이터(사이트) 이름을 의미
        # 경로 설정하려고 작성하는 것
        self.data_folder = 'Robotis'

        # 학습에 사용할 train.txt 파일 경로 작성 
        # ex) 'Robotis/train_data_2023_8_4_10/train.txt'
        self.image_sets = [('train_data_2023_8_4_10', 'train')] # 학습에 사용할 train.txt 파일 경로 작성 ex) 'robotis_thyssen/1/train.txt'

        # 학습에 사용할 val.txt 파일 경로 작성
        # ex) 'Robotis/train_data_2023_8_4_10/val.txt'
        self.test_sets = [('train_data_2023_8_4_10', 'val')]

        # ---------- transform config ------------ #
        # augmentation 적용 비율을 의미
        # 이미 기본 값이 정해져 있으므로, 주석처리해도 됨
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.0


        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]


    def create_cache_dataset(self, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform, VOCDetection
        self.cache_dataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "voc", self.data_folder),
            image_sets= self.image_sets,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            cache=True,
        )

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            VOCDetection,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            if self.cache_dataset is None:
                assert cache_img is None, "cache is True, but cache_dataset is None"
                dataset = VOCDetection(
                    data_dir=os.path.join(get_yolox_datadir(), "voc", self.data_folder),
                    image_sets= self.image_sets,
                    img_size=self.input_size,
                    preproc=TrainTransform(
                        max_labels=50,
                        flip_prob=self.flip_prob,
                        hsv_prob=self.hsv_prob),
                    cache=cache_img,
                )
            else:
                dataset = self.cache_dataset

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(
            len(self.dataset), seed=self.seed if self.seed else 0
        )

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import VOCDetection, ValTransform

        valdataset = VOCDetection(
            data_dir=os.path.join(get_yolox_datadir(), "voc", self.data_folder),
            image_sets=self.test_sets,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )
        return evaluator
