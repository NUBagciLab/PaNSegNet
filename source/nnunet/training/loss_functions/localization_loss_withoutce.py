from cProfile import label
from turtle import forward
import torch
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import GDL, SoftDiceLoss, SoftDiceLossSquared
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


class LogBarrierLoss(nn.Module):
    def __init__(self, t=0.1, bound_ratio=0.1):
        super().__init__()
        self.t_0 = t
        self.t = self.t_0
        self.delta_value = 1e-4
        self.bound_ratio = 0.1
        self.eps = 1e-5
        self.clip_value = 100
        self.lower_bound, self.up_bound = 1 - bound_ratio, 1+bound_ratio
    
    def forward(self, predict:torch.Tensor, target: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
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

        sum_target = torch.sum(y_onehot[:,1].flatten(2), dim=-1)
        sum_predict = torch.sum(predict[:,1].flatten(2), dim=-1)

        upper_bound = self.up_bound*sum_target
        lower_bound = self.lower_bound*sum_target

        lower_barrier = (lower_bound - sum_predict) / (sum_target +self.eps)
        up_barrier = (sum_predict - upper_bound) / (sum_target +self.eps)

        res = self.get_loss(lower_barrier) +self.get_loss(up_barrier)
        #print(res)
        res = torch.mean(torch.clip(res, min=-self.clip_value, max=self.clip_value))
        # print(res)
        return res

    def get_loss(self, z):
        determine = (z<=-1/self.t**2).to(torch.float32)
        left = determine*(-torch.log(torch.abs(z)) / self.t)
        right = (1-determine)*(self.t*z - np.log(1/self.t**2) / self.t + 1 / self.t)
        return left + right

    def set_t(self, dice):
        '''
        Set t value according to the dice loss
            add delta / (dice_loss)
        '''
        self.t = self.t_0*np.exp(1/self.t_0*dice)

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
        # self.l1_loss = nn.L1Loss()
        # self.alpha = 0.1

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

        for i in range(1, n_dim-2):
            if i == 1:
                dim_predict = predict[:,1:].flatten(3)
                dim_label = y_onehot[:,1:].flatten(3)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = dis_calculation(predict=dim_predict, target=dim_label)
            else:
                dim_predict = predict[:,1:].transpose(2, i+1)
                dim_label = y_onehot[:,1:].transpose(2, i+1)
                dim_predict = dim_predict.flatten(3)
                dim_label = dim_predict.flatten(3)
                dim_predict = torch.sum(dim_predict, dim=-1)
                dim_label = torch.sum(dim_label, dim=-1)
                loss = loss + dis_calculation(predict=dim_predict, target=dim_label)

        # print(l1_regularization)
        # loss = loss + self.alpha*l1_regularization
        return loss


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

        self.ignore_label = ignore_label

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_LC_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_LC_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.loc_loss = LocalizationLoss()
        self.regularization = LogBarrierLoss()
        self.ignore_label = ignore_label
        self.dice_coeff = 0

        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0

        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        lc_loss = self.loc_loss(net_output, target)
        reg_loss  = self.regularization(net_output, target)
        # print('lc_loss', lc_loss)
        # print('reg_loss', reg_loss)

        if self.aggregate == "sum":
            result = self.weight_dice * dc_loss + (self.dice_coeff**2)*(lc_loss+reg_loss)
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

    def set_dice(self, dice):
        '''
        set dice value for updating the loss weight
        '''
        self.dice_coeff = dice
        self.regularization.set_t(dice=dice)
