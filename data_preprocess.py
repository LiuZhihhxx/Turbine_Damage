import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from dataset import YoloDataset
# from model import YoloNet
# from loss import YoloLoss
# from trainer import Trainer
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from args import args_parser

# Introduce the hyper-parameters
args = args_parser()

num_classes = 5+1
batch_size = 8
num_epochs = 10
learning_rate = 2e-5
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


class MyDataset(Dataset):
    def __init__(self, img_folder, lbl_folder, transform=None):
        self.img_folder = img_folder
        self.lbl_folder = lbl_folder
        self.transform = transform
        self.labels = []

        # Load labels
        for file in os.listdir(lbl_folder):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(lbl_folder, file))
                root = tree.getroot()
                label = []
                for obj in root.iter('object'):
                    name = obj.find('name').text.lower().strip()
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text) - 1
                    ymin = int(bbox.find('ymin').text) - 1
                    xmax = int(bbox.find('xmax').text) - 1
                    ymax = int(bbox.find('ymax').text) - 1
                    label.append([xmin, ymin, xmax, ymax, name])
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, str(idx) + '.jpg')
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx])
        return img, label


def generate_dataloader():
    # Define paths to image and label folders
    train_img_folder = 'images/train'
    train_lbl_folder = 'labels/train'
    test_img_folder = 'images/val'
    test_lbl_folder = 'labels/val'

    # Define transformations for the images
    transform = transforms.Compose([
        transforms.Resize([416, 416]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 固定参数
    ])

    train_dataset = MyDataset(train_img_folder, train_lbl_folder, transform=transform)
    test_dataset = MyDataset(test_img_folder, test_lbl_folder, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader


