"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # 得到device
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        # feature需至少3维
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        # 大于3维则降维
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        # labels 和 mask 不能同时存在
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        # labels 不存在，则mask默认为 eye
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        # labels 存在，则通过labels生成mask
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 对比数量
        contrast_count = features.shape[1]
        # 切片后连接 --> n*b, d
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # 对比模式 1，只计算当前anchor和其他对比特征的损失
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        # 对比模式 全部，把所有都当作anchor计算损失后的和
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        """
        mask: b, b
        contrast_feature: n*b, d
        两类情况
        (1) anchor_count = 1
        anchor_feature: b, d
        (2) anchor_count = n
        anchor_feature: n*b, d
        """

        # 计算 logits = (anchor * contrast)/T
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases --> 1 - eye
        # scatter(self, dim, index, src)
        # self[index[i][j][k]][j][k] *= src[i][j][k]  # if dim == 0
        # self[i][index[i][j][k]][k] *= src[i][j][k]  # if dim == 1
        # self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2
        logits_mask = torch.scatter(
            torch.ones_like(mask), # b,n*b
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device), # b, 1
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        # 自监督的对比学习？只有单个 feature
        return loss
