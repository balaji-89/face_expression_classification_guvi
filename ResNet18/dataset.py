"""
Adapted from https://github.com/usef-kh/fer/tree/master/data
"""
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import os
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample

def load_data(path='fer2013.csv'):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

    return fer2013, emotion_mapping


def prepare_data():
    image_array = []
    image_label = []

    emotion_mapping = {'angry' : 0, 'disgust' : 1,'fear' : 2,'happy' : 3,'sad' : 4,'surprise' : 5, 'neutral' : 6}
    test_root_path = '/home/laptop-kl-11/personal_project/face_expression_classification/dataset/test'
    for emotion_dir in os.listdir(test_root_path):
        for img_name in os.listdir(os.path.join(test_root_path,emotion_dir)):
            img = cv2.imread(os.path.join(test_root_path,emotion_dir,img_name),cv2.IMREAD_GRAYSCALE)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_array.append(img)
            image_label.append(emotion_dir)
        


    image_label = list(map(lambda x: emotion_mapping[x], image_label))

    return np.array(image_array), np.array(image_label)



def get_dataloaders(path='datasets/fer2013/fer2013.csv', bs=64, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """


    xtest, ytest = prepare_data()
    mu, st = 0, 255
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    test = CustomDataset(xtest, ytest, test_transform)
    testloader = DataLoader(test, batch_size=64, shuffle=True)
    return testloader
