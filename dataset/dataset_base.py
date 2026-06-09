from pathlib import Path
import torch
from torch.utils import data as data
from dataset.utils import FileClient, imfrombytes, paired_random_crop_multi, img2tensor, scandir, read_img_seq, generate_frame_indices
import random
from os import path as osp
import glob

class ImageFusionDataset(data.Dataset):

    def __init__(self, opt):
        super(ImageFusionDataset, self).__init__()
        self.opt = opt
        self.ir_gt_root = Path(opt["dataroot_gt_ir"]) 
        self.vi_gt_root = Path(opt["dataroot_gt_vi"])
        # self.num_frame = opt["num_frame"]
        
        self.ir_images = sorted([f for f in self.ir_gt_root.glob("*.png")])
        self.vi_images = sorted([f for f in self.vi_gt_root.glob("*.png")])

        # 文件客户端 (I/O 后端)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False

        # self.interval_list = opt.get("interval_list", [1])
        # self.random_reverse = opt.get("random_reverse", False)

    def __getitem__(self, index):
        try:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

            gt_size = self.opt["gt_size"]
            # 获取当前图像对应的低清图像和高清图像路径
            ir_image_path = self.ir_images[index]
            vi_image_path = self.vi_images[index]
            
            # 获取图像文件的字节
            img_ir_bytes = self.file_client.get(ir_image_path, "gt")
            img_vi_bytes = self.file_client.get(vi_image_path, "gt")
            
            # 从字节中解码图像
            img_ir_gt = imfrombytes(img_ir_bytes, float32=True)
            img_vi_gt = imfrombytes(img_vi_bytes, float32=True)

            # 随机裁剪
            img_ir_gts, img_vi_gts = paired_random_crop_multi([img_ir_gt], [img_vi_gt], gt_size)
            # print()
        
            img_ir_results = img_ir_gts*2.0-1.0
            img_vi_results = img_vi_gts*2.0-1.0

            img_ir_results = img2tensor(img_ir_results)
            # print(torch.max(img_ir_results),torch.min(img_ir_results))
            img_vi_results = img2tensor(img_vi_results)

            # img_ir_gts = torch.stack(img_ir_results, dim=0)
            # img_vi_gts = torch.stack(img_vi_results, dim=0)

            return {"gt_ir": img_ir_results, "gt_vi": img_vi_results, "key": str(ir_image_path)}
        except Exception as e:
            print(f"[Dataset Error] Index {index} failed: {e}")
            dummy = torch.zeros(3, self.opt["gt_size"], self.opt["gt_size"])
            return {"gt_ir": dummy, "gt_vi": dummy, "key": "error_image"}

    def __len__(self):
        return len(self.ir_images)

class ImageTestDataset(data.Dataset):

    def __init__(self, opt):
        super(ImageTestDataset, self).__init__()
        self.opt = opt
        # 分别是高清图像和低清图像的路径
        self.ir_gt_root = Path(opt["dataroot_gt_ir"])
        self.vi_gt_root = Path(opt["dataroot_gt_vi"])
        
        # 获取所有图像路径
        self.ir_images = sorted([f for f in self.ir_gt_root.glob("*.png")])
        self.vi_images = sorted([f for f in self.vi_gt_root.glob("*.png")])

        # 文件客户端 (I/O 后端)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # 获取图像路径
        ir_image_path = self.ir_images[index]
        vi_image_path = self.vi_images[index]

        # 读取图像
        img_ir_bytes = self.file_client.get(ir_image_path, "gt")
        img_vi_bytes = self.file_client.get(vi_image_path, "gt")

        # 从字节解码图像
        img_ir_gt = imfrombytes(img_ir_bytes, float32=True)
        img_vi_gt = imfrombytes(img_vi_bytes, float32=True)
        img_ir_gt = img_ir_gt*2.0-1.0
        img_vi_gt = img_vi_gt*2.0-1.0

        # 不进行随机裁剪或翻转，直接返回
        img_ir_results = img2tensor([img_ir_gt])
        img_vi_results = img2tensor([img_vi_gt])

        # img_ir_gts = torch.stack(img_ir_results, dim=0)
        # img_vi_gts = torch.stack(img_vi_results, dim=0)

        return {"gt_ir": img_ir_results, "gt_vi": img_vi_results, "key": str(ir_image_path)}

    def __len__(self):
        return len(self.ir_images)