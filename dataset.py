#!/usr/bin/env python3
import os
import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
import random
from lib.data_preprocess.preprocess import prepare_train_input, prepare_test_input
from lib.util import load_config, my_collate
from torch.utils.data import DataLoader

class DeepfakeDataset(Dataset):
    r"""DeepfakeDataset Dataset.

    The folder is expected to be organized as followed: root/cls/xxx.img_ext

    Labels are indices of sorted classes in the root directory.

    Args:
        mode: train or test.
        config: hypter parameters for processing images.
    """

    def __init__(self, mode: str, config: dict):
        super().__init__()

        self.config = config
        self.mode = mode
        self.root = self.config['dataset']['img_path']
        self.landmark_path = self.config['dataset']['ld_path']
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.class_dict = self.collect_class()
        self.train_samples, self.val_samples, self.test_samples = self.collect_samples_divide() 

        if mode == 'train':
            self.samples = self.train_samples
        else:
            self.samples = self.test_samples + self.val_samples    

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def collect_samples(self) -> List:
        samples = []
        directory = os.path.expanduser(self.root)
        for key in sorted(self.class_dict.keys()):
            d = os.path.join(directory, key)
            if not os.path.isdir(d):
                continue
            for r, _, filename in sorted(os.walk(d, followlinks=True)):
                for name in sorted(filename):
                    path = os.path.join(r, name)
                    info_key = path[:-4]
                    video_name = '/'.join(path.split('/')[:-1])
                    info_meta = self.info_meta_dict[info_key]
                    landmark = info_meta['landmark']
                    class_label = int(info_meta['label'])
                    source_path = info_meta['source_path'] + path[-4:]
                    samples.append(
                        (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                    )

        return samples
    
    def gh_collect_samples(self) -> List:
        samples = []
        all_samples_from_json = self.info_meta_dict.keys()
        for key in all_samples_from_json:
            video_name = '/'.join(key.split('/')[:-1])
            info_meta = self.info_meta_dict[key]
            landmark = info_meta['landmark']
            class_label = int(info_meta['label'])
            source_path = info_meta['source_path'] + '.png'
            path = key+'.png'
            samples.append(
                        (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                    )
        return samples

    def collect_samples_divide(self) -> List:
        train_sample = []
        val_sample = []
        test_sample = []
        with open('train.json', 'r') as file:
            train_sample_list = json.load(file)
        with open('val.json', 'r') as file:
            val_sample_list = json.load(file)
        with open('test.json', 'r') as file:
            test_sample_list = json.load(file)


        for key in self.info_meta_dict.keys():
            video_num = key.split('/')[-2]
            video_name = '/'.join(key.split('/')[:-1])
            info_meta = self.info_meta_dict[key]
            landmark = info_meta['landmark']
            class_label = int(info_meta['label'])
            source_path = info_meta['source_path'] + '.png'
            path = key+'.png'
            subset_label = 1 # 1 for train , 2 val , 3 test

            if "Real" not in key:
                video_num_1,video_num_2 = video_num.split('_')
                for id1,id2 in train_sample_list:
                    if id1 == video_num_1 and id2 == video_num_2 or id2 == video_num_1 and id1 == video_num_2:
                        subset_label = 1
                for id1,id2 in val_sample_list:
                    if id1 == video_num_1 and id2 == video_num_2 or id2 == video_num_1 and id1 == video_num_2:
                        subset_label = 2
                for id1,id2 in test_sample_list:
                    if id1 == video_num_1 and id2 == video_num_2 or id2 == video_num_1 and id1 == video_num_2:
                        subset_label = 3
            else:
                for id1,id2 in train_sample_list:
                    if video_num == id1 or video_num == id2:
                        subset_label = 1
                for id1,id2 in val_sample_list:
                    if video_num == id1 or video_num == id2:
                        subset_label = 2
                for id1,id2 in test_sample_list:
                    if video_num == id1 or video_num == id2:
                        subset_label = 3
            
            if subset_label == 1:
                train_sample.append(
                    (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                )
            elif subset_label == 2:
                val_sample.append(
                    (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                )
            else:
                test_sample.append(
                    (path, {'labels': class_label, 'landmark': landmark,
                                'source_path': source_path,
                                'video_name': video_name})
                )
        return train_sample, val_sample, test_sample
    
     

        


    def collect_class(self) -> Dict:
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort(reverse=True)
        return {classes[i]: np.int32(i) for i in range(len(classes))}

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        source_path = label_meta['source_path']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        if img is None or source_img is None:
            print("这里读到的img或source_img图片是空，会让后续代码报错！！！")
            return None, None


        if self.mode == "train":
            img, label_dict = prepare_train_input(
                img, source_img, ld, label, self.config, self.do_train
            )
            if isinstance(label_dict, str):
                return None, label_dict

            location_label = torch.Tensor(label_dict['location_label'])
            confidence_label = torch.Tensor(label_dict['confidence_label'])
            img = torch.Tensor(img.transpose(2, 0, 1))
            return img, (label, location_label, confidence_label)

        elif self.mode == 'test':
            img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
            img = torch.Tensor(img[0].transpose(2, 0, 1))
            video_name = label_meta['video_name']
            return img, (label, video_name)

        else:
            raise ValueError("Unsupported mode of dataset!")

    def __len__(self):
        return len(self.samples)




class CelebDF(Dataset):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.rng = np.random
        #self.landmark_path = "/data1/msf/00YuPeipeng/dfdccelebdf/dfdc/test/ldm.json"
        self.landmark_path = '/home/gaohui/Celeb-DF_CADDM/ldm.json'
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.samples = self.gh_collect_samples()

    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict

    def gh_collect_samples(self) -> List:
        samples = []
        all_samples_from_json = self.info_meta_dict.keys()
        for key in all_samples_from_json:
            info_meta = self.info_meta_dict[key]
            landmark = info_meta['landmark']
            class_label = int(info_meta['label'])
            video_name = key.split('/')[-1]
            samples.append(
                (key, {'labels': class_label, 'landmark': landmark, 'video_name': video_name})
            )
        return samples

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        path = path + '.png'
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        if label == 0:
            label = 1
        else:
            label = 0
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print("这里读到的img或source_img图片是空，会让后续代码报错！！！")
            return None, None

        img, label_dict = prepare_test_input(
            [img], ld, label, self.config
        )
        img = torch.Tensor(img[0].transpose(2, 0, 1))
        video_name = label_meta['video_name']
        return img, (label, video_name)

    def __len__(self):
        return len(self.samples)




class DFDC(Dataset):
    def __init__(self, mode: str, config: dict):
        super().__init__()
        self.config = config
        self.mode = mode
        self.rng = np.random
        assert mode in ['train', 'test']
        self.do_train = True if mode == 'train' else False
        self.info_meta_dict = self.load_landmark_json(self.landmark_path)
        self.samples = self.gh_collect_samples()




    def load_landmark_json(self, landmark_json) -> Dict:
        with open(landmark_json, 'r') as f:
            landmark_dict = json.load(f)
        return landmark_dict
    


    def gh_collect_samples(self) -> List:
        samples = []
        all_samples_from_json = self.info_meta_dict.keys()
        for key in all_samples_from_json:
            
            info_meta = self.info_meta_dict[key]
            landmark = info_meta['landmark']
            class_label = int(info_meta['label'])
            video_name = key.split('/')[-2]
            samples.append(
                        (key, {'labels': class_label, 'landmark': landmark , 'video_name': video_name})
                    )
        return samples
    

    def __getitem__(self, index: int) -> Tuple:
        path, label_meta = self.samples[index]
        ld = np.array(label_meta['landmark'])
        label = label_meta['labels']
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None :
            print("这里读到的img或source_img图片是空，会让后续代码报错！！！")
            return None, None
        
        img, label_dict = prepare_test_input(
                [img], ld, label, self.config
            )
        img = torch.Tensor(img[0].transpose(2, 0, 1))
        video_name = label_meta['video_name']
        return img, (label, video_name)

    def __len__(self):
        return len(self.samples)





if __name__ == "__main__":
    
    config = load_config('./configs/caddm_train.cfg')
    train_set = DeepfakeDataset(mode="train", config=config)
    test_set = DeepfakeDataset(mode="test", config=config)
    print(len(train_set))
    print(len(test_set))
    




# vim: ts=4 sw=4 sts=4 expandtab
