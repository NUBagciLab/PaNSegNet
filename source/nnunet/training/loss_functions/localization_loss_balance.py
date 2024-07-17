from cProfile import label
from turtle import forward
import torch
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np


def dis_calculation(predict:torch.Tensor, target: torch.Tensor, eps:float=1e-5):
    '''
    Calculate the one dimension distance
        predict: [N, Channel, H]
        target: [N, Channel, H]
    '''
    n_length = predict.size(-1)

    dist_pred = torch.cumsum(predict, dim=-1) / (torch.sum(predict, dim=-1, keepdim=True) + eps)
    dist_target = torch.cumsum(target, dim=-1) / (torch.sum(target, dim=-1, keepdim=True) + eps)
    dim_loss = torch.mean(torch.abs(dist_pred-dist_target)) * torch.sqrt(torch.tensor(n_length, device=dist_pred.device))

    return dim_loss


class LocalizationLoss(nn.Module):
    '''
    The localization loss using one dimension wasserstein distance for image segmentation
    Background is not calculated
    Args:
        smooth: the smooth value
    '''
    def __init__(self, smooth:float=1e-5):
        super().__init__()
        self.smooth = smooth
        self.l1_loss = nn.L1Loss()
        self.alpha = 0.1

    def forward(self, predict:torch.Tensor, target: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        n_dim = predict.dim()
        shp_x = predict.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(predict.shape, target.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(shp_x)
                if predict.device.type == "cuda":
                    y_onehot = y_onehot.cuda(predict.device.index)
                y_onehot.scatter_(1, target, 1)
            
            predict_value = torch.argmax(predict, dim=1)
            predict_value = predict_value.unsqueeze_(1)
            predict_onehot = torch.zeros(shp_x)
            if predict.device.type == "cuda":
                predict_onehot = predict_onehot.cuda(predict.device.index)
            predict_onehot.scatter_(1, predict_value, 1)
            fp_index = (predict_onehot==1)&(y_onehot==0)
            fn_index = (predict_onehot==0)&(y_onehot==1)

        fp = predict * fp_index
        fn = y_onehot * fn_index

        for i in range(1, n_dim-2):
            if i == 1:
                dim_predict = fp[:,1:].flatten(3)
                dim_label = fn[:,1:].flatten(3)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = dis_calculation(predict=dim_predict, target=dim_label)
            else:
                dim_predict = fp[:,1:].transpose(2, i+1)
                dim_label = fn[:,1:].transpose(2, i+1)
                dim_predict = dim_predict.flatten(3)
                dim_label = dim_predict.flatten(3)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = loss + dis_calculation(predict=dim_predict, target=dim_label)

        # plus L1 regularization
        l1_regularization = self.l1_loss(torch.mean(fp[:,1:].flatten(2), dim=-1), -torch.mean(fn[:,1:].flatten(2), dim=-1))
        # print(l1_regularization)
        loss = loss + self.alpha*l1_regularization
        return loss



class DC_and_LC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, log_dice=False, ignore_label=None):
        super().__init__()
        self.dc_ce_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice, log_dice, ignore_label)
        self.loc_loss = LocalizationLoss()
        self.alpha = 0.01
    
    def forward(self, net_output, target):
        dc_loss = self.dc_ce_loss(net_output, target)
        lc_loss = self.loc_loss(net_output, target)
        loss = dc_loss + self.alpha*lc_loss
        return loss


class DC_and_LC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, log_dice=False, ignore_label=None):
        super().__init__()
        self.dc_ce_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs, aggregate, square_dice, weight_ce, weight_dice, log_dice, ignore_label)
        self.loc_loss = LocalizationLoss()
        self.alpha = 0.01
        self.beta = 0.1
    
    def forward(self, net_output, target):
        dc_loss = self.dc_ce_loss(net_output, target)
        lc_loss = self.loc_loss(net_output, target)
        loss = dc_loss + self.alpha*lc_loss
        return loss
