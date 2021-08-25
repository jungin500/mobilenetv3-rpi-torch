from datetime import datetime
from PIL import Image
import cv2
import glob
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import transforms
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from model import MobileNetV3_RPiWide

class LabelReader(object):
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        if 'pretrained' in label_file_path:
            print("INFO: Using Pretrained label list! (not custom one)")

    def load_label(self):
        label_map = {}
        # Read label file into label map
        if os.path.isfile(self.label_file_path):
            with open(self.label_file_path, 'r') as f:
                label_name_body = f.read().strip()
                label_name_lines = label_name_body.split("\n")
                for label_entry in tqdm(label_name_lines, desc='레이블 파일 읽기 작업'):
                    synset_name, label_name = label_entry.strip().split("|")
                    label_map[synset_name] = label_name

            print(f"레이블 파일 읽기 완료: 총 {len(list(label_map.keys()))}개 레이블 검색")
            return label_map
        else:
            return None

            
class ImageNet(torch.utils.data.Dataset):
    def __init__(self, labels, root_dir, pre_resize=None, transform=None):
        super(ImageNet, self).__init__()

        self.labels = labels
        self.pre_resize = pre_resize
        self.transform = transform

        self.img_path_list = []
        self.img_class_list = []
        self.load_list(root_dir)

    def load_list(self, root_dir):
        label_index = 0
        for label in tqdm(self.labels.keys(), desc='이미지 파일 리스트 읽기 작업'):
            item_dir = os.path.join(root_dir, label)
            file_list = glob.glob(item_dir + os.sep + "*.JPEG")
            self.img_path_list += file_list
            self.img_class_list += [label_index] * len(file_list)
            label_index += 1

        if len(self.img_path_list) != len(self.img_class_list):
            raise RuntimeError(f"이미지 데이터 {len(self.img_path_list)}개와 클래스 데이터 {len(self.img_class_list)}개가 서로 다릅니다!")

        print(f"총 {len(self.img_path_list)}개 이미지 리스트 데이터 및 실효 레이블 {len(list(set(self.img_class_list)))}개 로드 성공")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # PIL-version
        # image = Image.open(self.img_path_list[idx]).convert("RGB")
        # if self.transform is not None:
        #     image = self.transform(image)
        label = torch.Tensor([self.img_class_list[idx]]).type(torch.int64).squeeze(dim=0)
        image = cv2.imread(self.img_path_list[idx], cv2.IMREAD_COLOR)
        if self.pre_resize is not None:
            image = cv2.resize(image, self.pre_resize)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image / 255.0).astype(np.float32)
        if self.transform is not None:
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            image -= np.array([[[0.4547857]], [[0.4349471]], [[0.40525291]]])  # mean
            image /= np.array([[[0.12003352]], [[0.12323549]], [[0.1392444]]])  # stddev
            image = torch.from_numpy(image)
        return image, label


if __name__ == '__main__':
    lr=0.01
    load_checkpoint = None

    device = torch.device('cuda')

    # dataset
    labels = LabelReader('imagenet_subset.list').load_label()
    dataset = ImageNet(
        labels=labels,
        root_dir=r'C:\ILSVRC2012Subset',
        pre_resize=(480, 288),
        # transform=transforms.Compose([
        #     # transforms.Resize((480, 288)),
        #     # transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=[0.4547857, 0.4349471, 0.40525291],
        #         std=[0.12003352, 0.12323549, 0.1392444]
        #     )
        # ]),
    )

    train_dataset_items = int(len(dataset) * 0.9)
    val_dataset_items = len(dataset) - train_dataset_items
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_dataset_items, val_dataset_items])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # model
    if load_checkpoint is not None:
        model = MobileNetV3_RPiWide.load_from_checkpoint(load_checkpoint)
    else:
        model = MobileNetV3_RPiWide(width_mult=1.0, num_classes=200)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/',
        filename='mb3-rpi-lightning-fp16-epoch{epoch:02d}-acc{val_acc:.2f}',
        save_top_k=3,
        mode='max'
    )

    chechpoint_earlystopping_trainloss = EarlyStopping(
        monitor='train_loss',
        min_delta=1e-5,
        patience=3,
        mode='min'
    )
    
    chechpoint_earlystopping_valacc = EarlyStopping(
        monitor='val_acc',
        min_delta=0.01,
        patience=1,
        mode='max'
    )

    trainer = pl.Trainer(gpus=1, precision=16, limit_train_batches=0.1, callbacks=[checkpoint_callback, chechpoint_earlystopping_trainloss, chechpoint_earlystopping_valacc])
    trainer.fit(model, train_loader, val_loader)