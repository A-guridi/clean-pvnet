import torch.nn as nn
from lib.utils import net_utils
import torch


# added Focal loss for segmentation
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.vote_crit = torch.nn.functional.smooth_l1_loss
        self.seg_crit = nn.CrossEntropyLoss()
        # self.focal_crit = net_utils.FocalLoss()

    def forward(self, batch):
        output = self.net(batch['inp'])

        scalar_stats = {}
        loss = 0

        if 'pose_test' in batch['meta'].keys():
            loss = torch.tensor(0).to(batch['inp'].device)
            return output, loss, {}, {}

        weight = batch['mask'][:, None].float()
        # for the RGB output

        """
        vote_loss = self.vote_crit(output['vertex'] * weight, batch['vertex'] * weight, reduction='sum')
        vote_loss = vote_loss / weight.sum() / batch['vertex'].size(1)
        scalar_stats.update({'vote_loss': vote_loss})
        loss += vote_loss
        """

        # for the polarization output
        if True:
            vote_loss_pol = self.vote_crit(output['vertex_pol'] * weight, batch['vertex'] * weight, reduction='sum')
            vote_loss_pol = vote_loss_pol / weight.sum() / batch['vertex'].size(1)
            scalar_stats.update({'vote_loss_pol': vote_loss_pol})
            loss += vote_loss_pol

        # for the RBG
        """
        mask = batch['mask'].long()
        seg_loss = self.seg_crit(output['seg'], mask)
        scalar_stats.update({'seg_loss': seg_loss})
        loss += seg_loss
        """

        # for the polarization
        if True:
            seg_loss_pol = self.seg_crit(output['seg_pol'], mask)
            scalar_stats.update({'seg_loss_pol': seg_loss_pol})
            loss += seg_loss_pol

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats
