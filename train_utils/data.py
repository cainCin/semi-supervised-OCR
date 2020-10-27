
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
import imgaug as ia
import glob
from PIL import Image
import torch
import os

class AlignedDataset(Dataset):
    def __init__(self, data_path, transform=None, direction="AtoB", additional_transform=None):
        self.load_data(data_path)
        self.transform = transform
        self.additional_transform = additional_transform
        direction_list = ["AtoB", "BtoA"]
        assert direction in direction_list, print(f"direction not supported, only in {direction_list}")
        self.direction = direction
        pass

    def load_data(self, data_path):
        self.data = glob.glob(data_path + "/*")
        

    def __getitem__(self, index):
        data_path = self.data[index]
        AB = Image.open(data_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        if self.additional_transform is not None:
            # perform additional tranform to train image
            A = self.additional_transform(A)

        if self.transform is not None:
            # perform transform
            seed = np.random.randint(1e5)
            random.seed(seed)
            A = self.transform(A)
            random.seed(seed)
            B = self.transform(B)
        
        if self.direction == "AtoB":
            return A, B
        else:
            return B, A


    def __len__(self):
        return len(self.data)

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.load_data(data_path)
        self.transform = transform
        pass

    def load_data(self, data_path):
        self.data = glob.glob(data_path + "/*")
        

    def __getitem__(self, index):
        data_path = self.data[index]
        try:
            A = Image.open(data_path).convert('RGB')
        except:
            print(f"Error in loading {data_path}")
            A = Image.fromarray(np.zeros(shape=(1024,1024,3), dtype=np.uint8))

        if self.transform is not None:
            seed = np.random.randint(1e5)
            random.seed(seed)
            A = self.transform(A)

        return A


    def __len__(self):
        return len(self.data)

class Cinnamon_Layout(Dataset):
    def __init__(self, data_path, transform=None, target_transform=None):
        if isinstance(data_path, str):
            self.load_data(data_path)
        else:
            self.data = data_path
        
        self.transform = transform
        self.target_transform = target_transform
        pass

    def load_data(self, data_path):
        all_images = glob.glob(data_path + "/**/*", recursive=True)
        input_images = [image_name for image_name in all_images 
                            if "_GT" not in image_name \
                                and image_name.split(".")[-1].lower() in ["jpg", "jpeg", "png"]]
        label_images = [[".".join(image_name.split(".")[:-1]) + "_GT%d.jpg" %(i) for i in [0, 1, 2]]
                        for image_name in input_images]
        
        self.data = [input_images, label_images]

        

    def __getitem__(self, index):
        input_ = Image.open(self.data[0][index]).convert('L')
        label_ = [Image.open(label_path) for label_path in self.data[1][index]]

        if self.transform is not None:
            seed = np.random.randint(1e5)
            random.seed(seed)
            input_ = self.transform(input_)


        if self.target_transform is not None:
            transform_label = []
            for label in label_:
                random.seed(seed)
                transform_label.append(self.target_transform(label))
            
            label_ = torch.cat(transform_label, 0)

        # input_ = np.array(input_)
        # label_ = np.stack(label_, axis=-1)

        return input_, label_


    def __len__(self):
        return len(self.data[1])

class Cinnamon_OCR(Dataset):
    def __init__(self, data_path, root='', transform=None):
        self.load_data(data_path)
        self.root = root
        self.transform = transform
        pass

    def load_data(self, data_txt):
        with open(data_txt, "r", encoding="utf-8") as f:
            out = f.readlines()
            self.data = [item.split("|") for item in out]

    def __getitem__(self, index):
        image_path, label_text = self.data[index]
        image = Image.open(os.path.join(self.root, image_path)).convert('L')
        image = np.array(image)
        if self.transform:
            image = self.transform(image)

        return image, label_text

    def __len__(self):
        return len(self.data)



if __name__ == "__main__":
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    # # TODO: test Cinnamon Layout
    # datapath = r"D:\Workspace\cinnamon\data\layout\Invoice_Train"
    # input_transform = transforms.Compose([transforms.Resize(286),
    #                             transforms.RandomCrop(256),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor(),
    #                             # transforms.Normalize((0.5,), (0.5,))
    #                             ])

    # label_transform = transforms.Compose([transforms.Resize(286),
    #                             transforms.RandomCrop(256),
    #                             transforms.RandomHorizontalFlip(),
    #                             transforms.ToTensor(),
    #                             # transforms.Normalize((0.5,), (0.5,))
    #                             ])

    # dataset = Cinnamon_Layout(datapath, transform=input_transform, target_transform=label_transform)

    

    # dataset = Cinnamon_Layout(datapath, transform=None, target_transform=None)
    # dataloader = DataLoader(dataset, batch_size=2,
    #                             shuffle=True, num_workers=4)

    # for data, label in dataloader:
    #     print(data.shape, label.shape)
    #     break

    #TODO: test Cinnamon OCR
    datapath = r"D:\Workspace\cinnamon\data\ocr\showa_mini_1k2\showa_mini_1k2\showa_mini_1k_train.txt"
    dataroot = r"D:\Workspace\cinnamon\data\ocr\showa_mini_1k2\showa_mini_1k2"
    transform = transforms.Compose([
                                transforms.ToTensor()
                                ])
    dataset = Cinnamon_OCR(datapath, root=dataroot, transform=None)

    for image, label_text in dataset:
        print(image.shape, label_text)
        break
    pass