from turtle import forward
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np

from nnunet.utilities.tensor_utilities import sum_tensor
from .localization_loss_withoutdice import LocalizationLoss
from wSegLoss.sum_func.torchsum_funcx import (sum_two_func,
                                              diag_two_func)


class LogBarrierLoss(nn.Module):
    def __init__(self, t_0=1, mu=0.05, bound_ratio=0.1):
        super().__init__()
        self.t_0 = t_0
        self.mu = mu
        self.t = self.t_0
        self.delta_value = 1e-4
        self.bound_ratio = 0.1
        self.eps = 1e-4
        self.clip_value = 100
        self.lower_bound, self.up_bound = 1 - bound_ratio, 1+bound_ratio
    
    def forward(self, predict:torch.Tensor, target: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, volume]
            target: size [N, channel, volume]
        '''
        upper_bound = self.up_bound*target
        lower_bound = self.lower_bound*target

        lower_barrier = (lower_bound - predict) 
        up_barrier = (predict - upper_bound)

        res = self.get_loss(lower_barrier) +self.get_loss(up_barrier)
        res = torch.mean(torch.clip(res, min=-self.clip_value, max=self.clip_value))
        return res

    def get_loss(self, z):
        with torch.no_grad():
            determine = (z<=-1/self.t**2)
        left = determine*(-torch.log(torch.abs(z) + self.eps) / self.t)
        right = (~determine)*(self.t*z - np.log(1/self.t**2) / self.t + 1 / self.t)
        
        return left + right

    def up_epoch(self):
        '''
        Add t value according to training epoch
        '''
        self.t = self.t + self.mu


class NaiveDice(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1.0
    
    def forward(self, net_output, y_onehot):
        axes = tuple(range(2, len(net_output.size())))
        intersect = net_output* y_onehot
        # values in the denominator get smoothed
        denominator = net_output ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator
        dc = dc.mean()
        return -dc


def get_dist(func, input, eps):
    input_x, input_y = func(input)
    input_x, input_y = input_x / input_x.size(-1), input_y /input_x.size(-1)
    volume = torch.sum(input_x, dim=-1) + eps
    volume = volume.unsqueeze_(-1) 
    dist_x = torch.cumsum(input_x, dim=-1) / volume
    dist_y = torch.cumsum(input_y, dim=-1) / volume
    return dist_x, dist_y, volume.squeeze(-1)

class WSeg2DLoss(nn.Module):
    r'''
    Wassersterin Distance based Segmentation Loss
    Note: H should be equal to the W for distance calculation
    Args:
        is_xy (bool): calculate the x, y two dimension or x, y, xy, yx four dimension
        div_pes (float): float value for dividing
        alpha (float) :balance the weight between the distribution and log barrier
        log_paras (dict): the parameter dictionary for the log barrier
    '''
    def __init__(self, is_xy:bool=False, div_eps:float=0.5,
                       alpha:float=1, log_paras:Optional[dict]=None):
        super().__init__()
        self.div_eps = div_eps
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()
        self.cross_entro = nn.CrossEntropyLoss()
        if log_paras is not None:
            self.reg_loss = LogBarrierLoss(**log_paras)
        else:
            self.reg_loss = LogBarrierLoss()

        self.func_ops = [lambda input, eps: get_dist(sum_two_func, input, eps)]
        self.weights =[1]
        if is_xy:
            self.func_ops.append(lambda input, eps: get_dist(diag_two_func, input, eps))
            self.weights.append(np.sqrt(1/2))
    
    def forward(self, predict, target):
        '''
        Shape:
            - predict [N, C, H, W]
            - target [N, 1 or C, H, W]
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

        dist_loss_sum = 0
        predict, y_onehot = predict, y_onehot
        # positive part
        for funcs, weight in zip(self.func_ops, self.weights):
            dist_px, dist_py, volume_p = funcs(predict*y_onehot, self.div_eps)
            dist_tx, dist_ty, volume_t = funcs(y_onehot, self.div_eps)
            dist_loss = self.l1_loss(dist_px, dist_tx) + \
                            self.l1_loss(dist_py, dist_ty)
            dist_loss_sum = dist_loss_sum + dist_loss * weight
        # negative part
        for funcs, weight in zip(self.func_ops, self.weights):
            dist_px, dist_py, volume_p = funcs((1-predict)*(1-y_onehot), self.div_eps)
            dist_tx, dist_ty, volume_t = funcs(1-y_onehot, self.div_eps)
            dist_loss = self.l1_loss(dist_px, dist_tx) + \
                            self.l1_loss(dist_py, dist_ty)
            dist_loss_sum = dist_loss_sum + dist_loss * weight

        # regular_loss = self.reg_loss(volume_p, volume_t)
        regular_loss = self.l1_loss(volume_p, volume_t)
        # regular_loss = self.cross_entro(predict, target[:, 0].long())

        loss = dist_loss_sum + self.alpha * regular_loss
        '''
        print('dist loss', dist_loss_sum)
        print('regular_loss', regular_loss)
        print('total_loss', loss)
        '''
        return loss

    def up_epoch(self):
        self.reg_loss.up_epoch()


class WSeg3DLoss(WSeg2DLoss):
    '''
    Wassersterin Distance based Segmentation Loss for 3D
    Only difference is the Depth will be flatten to Channel
    '''
    def __init__(self, is_xy: bool = True, div_eps: float = 0.5,
                      alpha: float = 0.1, log_paras: Optional[dict] = None):
        super().__init__(is_xy, div_eps, alpha, log_paras)
        self.loc_loss = LocalizationLoss()
    
    def forward(self, predict, target):
        '''
        Note the format of nnUNet is h,w,d
        '''
        # print('org', predict.shape)
        shp_x = predict.shape
        shp_y = target.shape
        predict = torch.softmax(predict, dim=1)

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
    
        predict = torch.mean(predict, dim=2, keepdim=False)
        target = torch.mean(y_onehot, dim=2, keepdim=False)
        # print('after', predict.shape)
        loss = super().forward(predict, target)
        return loss


class CombinedWSegDice(WSeg2DLoss):
    '''
    Wassersterin Distance based Segmentation Loss for 3D
    Only difference is the Depth will be flatten to Channel
    '''
    def __init__(self, is_xy: bool = True, div_eps: float = 0.5,
                      alpha: float = 0.1, log_paras: Optional[dict] = None):
        super().__init__(is_xy, div_eps, alpha, log_paras)
        self.dice_loss = NaiveDice()
        self.cross_entro = nn.CrossEntropyLoss()
        self.epochs = 0
        self.beta = 0.
        self.mu = 0.001
    
    def forward(self, predict, target):
        '''
        Note the format of nnUNet is h,w,d
        '''
        # print('org', predict.shape)
        shp_x = predict.shape
        shp_y = target.shape
        ce_loss = self.cross_entro(predict, target[:, 0].long())
        predict = torch.softmax(predict, dim=1)
       
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
        dc_loss = self.dice_loss(predict, y_onehot)
        predict = torch.mean(predict, dim=2, keepdim=False)
        target = torch.mean(y_onehot, dim=2, keepdim=False)
        # print('after', predict.shape)
        # wseg_loss = super().forward(predict, target)
        loss = dc_loss + ce_loss
        return loss

    def up_epoch(self):
        self.epochs = self.epochs + 1
        self.beta = max(0, self.epochs*self.mu - 1)


if __name__ == '__main__':
    # Simple Test
    seg_loss = WSeg3DLoss(alpha=0.1)
    nbatch, channel, depth, height, width = 8, 16, 32, 128, 128
    predict = torch.randn(size=(nbatch, channel, depth, height, width),
                          device=torch.device('cuda'),
                          requires_grad=True)
    target = torch.ones(size=(nbatch, 1, depth, height, width),
                        device=torch.device('cuda'),
                        requires_grad=True)
    predict, target = torch.softmax(predict, dim=1), torch.softmax(target, dim=1)
    loss = seg_loss(predict, target)
    loss.backward()
    print(loss)

    volume1 = torch.mean(predict, dim=[3, 4])
    volume2 = torch.mean(target, dim=[3, 4])
    print(torch.mean(torch.abs(volume1-volume2)))
