import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class YoloModel(nn.Module):
    def __init__(self, num_classes):
        super(YoloModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = models.darknet53(pretrained=True)
        self.head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv1 = nn.Conv2d(1024, 255, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(512, 255, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(256, 255, kernel_size=1, stride=1, padding=0, bias=True)

        # Load pre-trained weights
        pretrained_dict = torch.load('yolov3.weights', map_location=device)  # 'cuda'
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def forward(self, x):
        # Backbone
        x = self.backbone(x)

        # Head
        x = self.head(x)

        # Detection
        pred = []
        for i, x_i in enumerate(x):
            if i > 0:
                x_i = torch.cat([x_i, x_prev], dim=1)
            x_prev = self.conv1(x_i)
            if i < 2:
                x_i = self.conv2(x_i)
                x_prev = nn.functional.interpolate(x_prev, scale_factor=2, mode='nearest')
            pred.append(x_prev.sigmoid())

        return pred
