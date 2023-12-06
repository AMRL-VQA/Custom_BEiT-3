# -----------------------------
# File: custom_dataset.py
# Author: Joon Cho
# Date: 2023-10-11
# Description:
# -----------------------------

import os
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from randaug import RandomAugment
from transformers import XLMRobertaTokenizer

current_path = os.path.dirname(os.path.realpath(__file__))
custom_dataset_path = os.path.join(current_path, 'custom_dataset')
default_jsonl_path = os.path.join(custom_dataset_path, 'custom.vqa.test.jsonl')
tokenizer_path = os.path.join(current_path,'model','beit3.spm')
default_tokenizer = XLMRobertaTokenizer(tokenizer_path)

def build_transform(is_train, img_size):
    if is_train:
        t = transforms.Compose([
            RandomResizedCropAndInterpolation(img_size, scale=(0.5, 1.0), interpolation='bicubic'), 
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                    2, 7, isPIL=True, 
                    augs=[
                        'Identity','AutoContrast','Equalize','Brightness','Sharpness', 
                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 
                    ]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])
    else:
        t = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

    return t

class CustomVQADataset(Dataset):
    def __init__(self, jsonl_path = default_jsonl_path, tokenizer = default_tokenizer, img_size=480, is_train=False, **kwargs):
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.is_train = is_train


        ans2label_file = os.path.join(custom_dataset_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = i
                label2ans.append(ans)

        self.ans2label = ans2label
        self.label2ans = label2ans

        if self.jsonl_path == default_jsonl_path and is_train:
            self.jsonl_path = os.path.join(custom_dataset_path,'custom.vqa.train.jsonl')
        
        self.dataframe = pd.read_json(path_or_buf=self.jsonl_path, lines=True)

        self.transform = build_transform(self.is_train, self.img_size)
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        img_name = os.path.join(custom_dataset_path, row['image_path'])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question']
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.is_train:
            answer = row['answer']
            try:
                label = self.ans2label[answer]
                one_hots = torch.nn.functional.one_hot(label, num_classes=3129)
            except KeyError:    # 3129개 이외의 클래스에 해당하는 답변 예외 처리
                one_hots = torch.tensor([0]*3129)

            return {
                'image': image.squeeze(),
                'question': question['input_ids'].squeeze(),
                'padding_mask': question['attention_mask'].squeeze().logical_not().to(int),
                'answer': one_hots.squeeze()
            }

        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
                'padding_mask': question['attention_mask'].squeeze().logical_not().to(int)
            }