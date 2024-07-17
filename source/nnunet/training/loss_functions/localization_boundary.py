import torch
from torch import nn
from torch.nn import functional as F

from nnunet.utilities.tensor_utilities import sum_tensor

import numpy as np


def get_boudary(mask:torch.Tensor, kernel_size=3):
    '''
    Calculate the pixel level boundary of one given mask
    Args:
        mask: [N, C, H, W, D]
    returns:
        boundary: [N, C, H, W, D]
    '''
    min_squeeze = F.max_pool3d(-mask, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    max_squeeze = F.max_pool3d(min_squeeze, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    boundary = F.relu(max_squeeze - min_squeeze)
    return boundary

def dis_calculation(predict:torch.Tensor, target: torch.Tensor, eps:float=0.5):
    '''
    Calculate the one dimension distance
        predict: [N, Channel, H, W]
        target: [N, Channel, H, W]
    '''
    dist_pred = torch.cumsum(predict, dim=-1) / (torch.sum(predict, dim=-1, keepdim=True) + eps)
    dist_target = torch.cumsum(target, dim=-1) / (torch.sum(target, dim=-1, keepdim=True) + eps)
    dim_loss = torch.mean(torch.abs(dist_pred-dist_target))
    return dim_loss


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
    
    def forward(self, predict:torch.Tensor, y_onehot: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        sum_target = torch.mean(y_onehot[:,1].flatten(2), dim=-1)
        sum_predict = torch.mean(predict[:,1].flatten(2), dim=-1)

        upper_bound = self.up_bound*sum_target
        lower_bound = self.lower_bound*sum_target

        lower_barrier = (lower_bound - sum_predict) 
        up_barrier = (sum_predict - upper_bound)

        res = self.get_loss(lower_barrier) +self.get_loss(up_barrier)
        #print(res)
        res = torch.mean(torch.clip(res, min=-self.clip_value, max=self.clip_value))
        # print(res)
        return res

    def get_loss(self, z):
        with torch.no_grad():
            determine = (z<=-1/self.t**2)
        left = determine*(-torch.log(torch.abs(z) + self.eps) / self.t)
        right = (~determine)*(self.t*z - np.log(1/self.t**2) / self.t + 1 / self.t)
        
        return left + right

    def set_t(self):
        '''
        Update the t value after each epoch
        '''
        self.t = self.t + self.mu


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

    def forward(self, predict:torch.Tensor, y_onehot: torch.Tensor):
        '''
        Args:
            predict: size [N, channel, H, W, D..]
            target: size [N, channel, H, W, D...]
        '''
        n_dim = predict.dim()

        for i in range(1, n_dim-2):
            if i == 1:
                dim_predict = predict[:,1:]
                dim_label = y_onehot[:,1:]
                dim_predict = torch.mean(dim_predict, dim=(-1, -2))
                dim_label = torch.mean(dim_label, dim=(-1, -2))
                loss = dis_calculation(predict=dim_predict, target=dim_label)

            else:
                dim_predict = predict[:,1:].transpose(2, i+1)
                dim_label = y_onehot[:,1:].transpose(2, i+1)
                dim_predict = torch.mean(dim_predict, dim=(-1, -2))
                dim_label = torch.mean(dim_label, dim=(-1, -2))
                loss = loss + dis_calculation(predict=dim_predict, target=dim_label)

        return loss


def get_tp_fp_fn_tn(net_output, y_onehot, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y_onehot, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


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
        self.loc_loss = LocalizationLoss()
        self.regularization = LogBarrierLoss()
        self.dc = SoftDiceLoss(apply_nonlin=None, **soft_dice_kwargs)

        self.ignore_label = ignore_label
        self.dice_coeff = 0

    def forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        net_output = torch.softmax(net_output, dim=1)
        shp_x = net_output.shape
        shp_y = target.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, target.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, target, 1)

            boundary_target = get_boudary(y_onehot)

        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        out_boundary = get_boudary(net_output)
        lc_loss = self.loc_loss(out_boundary, boundary_target)
        reg_loss  = self.regularization(out_boundary, boundary_target)
        # dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0

        # print('lc_loss', lc_loss)
        # print('reg_loss', reg_loss)

        if self.aggregate == "sum":
            result = lc_loss + 0.1*reg_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result

    def set_dice(self, dice):
        '''
        set dice value for updating the loss weight
        '''
        self.dice_coeff = dice
        self.regularization.set_t()
