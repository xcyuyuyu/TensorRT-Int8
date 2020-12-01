import pickle
import os
import numpy as np 
import torchvision.transforms as transforms
import torch

class CIFAR10_PARSE():

    base_folder = 'cifar-10-batches-py'
    
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root):
        self.data = []
        self.targets = []

        # if self.train:
        downloaded_list = self.train_list
        # else:
        # downloaded_list = self.test_list

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.idx = 0


    def __len__(self):
        return len(self.data)

    def get_batch_images(self, current_idx, batch_size):
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        imgs = np.zeros((batch_size, 3, 32, 32), dtype=np.float32)
        for i in range(batch_size):
            img, _ = self.get_one_image(current_idx + i)

            imgs[i,:,:,:] = transform(img).numpy()


        # import pdb;pdb.set_trace()
        return imgs


    def get_one_image(self, idx=None):

        if idx:
            self.idx = idx 
        
        if self.idx >= len(self):
            self.idx = 0

        img, target = self.data[self.idx], self.targets[self.idx]

        # self.idx += 1

        return img, target

    def get_one_image_torch(self, idx=None):

        img, target = self.get_one_image(idx)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        img = transform(img)
        # import pdb;pdb.set_trace()
        return img.unsqueeze(0), target
    
    
    def get_one_image_trt(self, pagelocked_buffer, idx=None):

        img, target = self.get_one_image(idx)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        img = transform(img).view(-1).numpy()

        np.copyto(pagelocked_buffer, img)



