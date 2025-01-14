from torch import nn
import torch
from torch.nn import functional as F
from .resnet import resnet18
from lib.csrc.ransac_voting.ransac_voting_gpu import ransac_voting_layer, ransac_voting_layer_v3, \
    estimate_voting_distribution_with_mean
from lib.config import cfg


# apparently, this class is also used for creating the network
class Resnet18(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32,
                 input_channels=3 + len(cfg.train.stokes_params), concat_polarization=True):
        super(Resnet18, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        # note that this was changed to adapt the input of the channels
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=8,
                               remove_avg_pool_layer=True)

        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.concat_pol = concat_polarization
        self.input_channels = input_channels

        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s
        # Here we differentiate between two kinds of layers, the ones
        # who concatenate the input from polarization too, and the ones who only take the RGB input
        # x8s->128, hence we multiply it by 2 because we combine x8s with x8s_pol in the polarized part
        self.conv8s_pol = nn.Sequential(
            nn.Conv2d(128 * 2 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up8sto4s = nn.UpsamplingBilinear2d(scale_factor=2)
        # x4s->64
        self.conv4s_pol = nn.Sequential(
            nn.Conv2d(64 * 2 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )

        # x2s->64
        self.conv2s_pol = nn.Sequential(
            nn.Conv2d(64 * 2 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        self.up4sto2s = nn.UpsamplingBilinear2d(scale_factor=2)

        # modified here s2dim+input_channels
        self.convraw_pol = nn.Sequential(
            nn.Conv2d(self.input_channels + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1)
        )
        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def decode_keypoint(self, output):
        vertex = output['vertex'].permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2 // 2, 2)
        mask = torch.argmax(output['seg'], 1)
        if cfg.test.un_pnp:
            mean = ransac_voting_layer_v3(mask, vertex, 512, inlier_thresh=0.99)
            kpt_2d, var = estimate_voting_distribution_with_mean(mask, vertex, mean)
            output.update({'mask': mask, 'kpt_2d': kpt_2d, 'var': var})
        else:
            kpt_2d = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
            output.update({'mask': mask, 'kpt_2d': kpt_2d})

    def forward(self, x, feature_alignment=False):
        # x_rgb is just the normal RGB encoder while xfc is the combination of both feature maps
        x2s, x4s, x8s, x16s, x32s, x_rgb, xfc, x2s_pol, x4s_pol, x8s_pol, x16s_pol, x32s_pol = self.resnet18_8s(x)

        ret = {}

        # used during training for RGB inference or always (train+inference) for polarized models
        if self.training or cfg.train.pol_inference:
            # on training, xfc is the combination of both feature maps from RGB and polarization
            fm = self.conv8s_pol(torch.cat([F.normalize(xfc), F.normalize(x8s), F.normalize(x8s_pol)], 1))
            fm = self.up8sto4s(fm)
            if fm.shape[2] == 136:
                fm = nn.functional.interpolate(fm, (135, 180), mode='bilinear', align_corners=False)

            fm = self.conv4s_pol(torch.cat([F.normalize(fm), F.normalize(x4s), F.normalize(x4s_pol)], 1))
            fm = self.up4sto2s(fm)
            fm = self.conv2s_pol(torch.cat([F.normalize(fm), F.normalize(x2s), F.normalize(x2s_pol)], 1))
            fm = self.up2storaw(fm)
            out_pol = self.convraw_pol(
                torch.cat([F.normalize(fm), F.normalize(x)], 1))  # this layer was changed depending on the channels of
            # input

            xfc = x_rgb
            x, _ = torch.split(x, [3, self.input_channels-3], dim=1)
            # when training, we change the value of xfc once used so the RGB decoder (conv8s) gets only
            # the RGB input. On testing, xfc will automatically be the output of the RGB encoder only

        # only used for non polarized models RGB only inference, used both during training and inference
        if not cfg.train.pol_inference:
            fm_rgb = self.conv8s(torch.cat([F.normalize(xfc), F.normalize(x8s)], 1))
            fm_rgb = self.up8sto4s(fm_rgb)
            if fm_rgb.shape[2] == 136:
                fm_rgb = nn.functional.interpolate(fm_rgb, (135, 180), mode='bilinear', align_corners=False)
            fm_rgb = self.conv4s(torch.cat([F.normalize(fm_rgb), F.normalize(x4s)], 1))
            fm_rgb = self.up4sto2s(fm_rgb)
            fm_rgb = self.conv2s(torch.cat([F.normalize(fm_rgb), F.normalize(x2s)], 1))
            fm_rgb = self.up2storaw(fm_rgb)
            out_rgb = self.convraw(torch.cat([F.normalize(fm_rgb), F.normalize(x)], 1))

            # we have now 2 outputs from 2 decoders, which will be fed to two different PnP algorithms and their
            # Losses added
            seg_pred_rgb = out_rgb[:, :self.seg_dim, :, :]
            ver_pred_rgb = out_rgb[:, self.seg_dim:, :, :]

            ret.update({'seg': seg_pred_rgb, 'vertex': ver_pred_rgb})

        # if training the polarized inference model, we use only the output from the combined pipeline
        else:
            seg_pred_pol = out_pol[:, :self.seg_dim, :, :]
            ver_pred_pol = out_pol[:, self.seg_dim:, :, :]
            ret.update({'seg': seg_pred_pol, 'vertex': ver_pred_pol})

        # if training the RGB inference model, we add the results to the output
        if self.training and not cfg.train.pol_inference:
            seg_pred_pol = out_pol[:, :self.seg_dim, :, :]
            ver_pred_pol = out_pol[:, self.seg_dim:, :, :]
            ret.update({'seg_pol': seg_pred_pol, 'vertex_pol': ver_pred_pol})

        if not self.training:
            with torch.no_grad():  # this is because the PnP algorithm is not differentiable -> no gradient
                self.decode_keypoint(ret)

        return ret


def get_res_pvnet(ver_dim, seg_dim):
    model = Resnet18(ver_dim, seg_dim)
    return model
