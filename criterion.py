import torch.nn as nn
from args import args_parser

args = args_parser()


class YOLOLoss(nn.Module):
    def __init__(self, S, B, C):
        super(YOLOLoss, self).__init__()
        self.S = 32  # grid size, = input image size/stride. For darknet-53 it's fixed as 32.
        self.B = 3  # number of bounding boxes per grid cell. In YOLO v3 it's fixed as 3.
        self.C = args.number_classes  # number of classes
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Compute the YOLO loss function.

        predictions : (N, S, S, Bx5+C) tensor
            The model's predictions for each image in the batch.
        targets : (N, S, S, Bx5+C) tensor
            The target labels for each image in the batch.

        Returns
        -------
        loss : scalar tensor
            The computed loss value.
        """
        # Split predictions and targets into location, size, confidence, and class components
        pred_loc = predictions[..., :2*self.B].reshape(-1, self.S, self.S, self.B, 2)
        pred_size = predictions[..., 2*self.B:4*self.B].reshape(-1, self.S, self.S, self.B, 2)
        pred_conf = predictions[..., 4*self.B:5*self.B].reshape(-1, self.S, self.S, self.B, 1)
        pred_cls = predictions[..., 5*self.B:].reshape(-1, self.S, self.S, self.C)

        true_loc = targets[..., :2*self.B].reshape(-1, self.S, self.S, self.B, 2)
        true_size = targets[..., 2*self.B:4*self.B].reshape(-1, self.S, self.S, self.B, 2)
        true_conf = targets[..., 4*self.B:5*self.B].reshape(-1, self.S, self.S, self.B, 1)
        true_cls = targets[..., 5*self.B:].reshape(-1, self.S, self.S, self.C)

        # Compute localization loss
        loc_loss = self.mse_loss(pred_loc, true_loc)

        # Compute size loss
        size_loss = self.mse_loss(pred_size.sqrt(), true_size.sqrt())

        # Compute confidence loss
        conf_loss = self.bce_loss(pred_conf, true_conf)

        # Compute class loss
        cls_loss = self.bce_loss(pred_cls, true_cls)

        # Compute total loss
        loss = loc_loss + size_loss + conf_loss + cls_loss

        return loss