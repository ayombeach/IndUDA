import torch

from . import resnet
from .domain_specific_module import BatchNormDomain
from utils import utils
from . import utils as model_utils
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from model.Inv_Model import InvNet, subnet_constructor
from torch.autograd import Function, Variable
backbones = [resnet]


class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DANet(nn.Module):
    def __init__(self, num_classes, feature_extractor='resnet50',
                 fx_pretrained=True, fc_hidden_dims=[], frozen=[],
                 num_domains_bn=2, dropout_ratio=(0.5,), block_num=5):
        super(DANet, self).__init__()
        self.feature_extractor = utils.find_class_by_name(
               feature_extractor, backbones)(pretrained=fx_pretrained,
               frozen=frozen, num_domains=num_domains_bn)

        self.block_num = block_num
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
        feat_dim = self.feature_extractor.out_dim
        self.in_dim = feat_dim

        self.FC = nn.Sequential()
        self.dropout = nn.Sequential()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                      else 0.0
            self.FC.add_module(str(k)+'D', nn.Dropout(p=cur_dropout_ratio))
            out_dim = fc_hidden_dims[k]
            self.FC.add_module(str(k)+'N', FC_BN_ReLU_Domain(in_dim, out_dim,
                  num_domains_bn))
            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
                  if self.num_hidden_layer < len(dropout_ratio) else 0.0

        self.FC.add_module('f_D', nn.Dropout(p=cur_dropout_ratio))
        # self.classifier = nn.Linear(int(feat_dim/2), num_classes)
        # self.domain_classifier = nn.Linear(int(feat_dim/2), 2)
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.INNet = InvNet(int(feat_dim/2), feat_dim, subnet_constructor, block_num=self.block_num)
        for key in range(len(self.FC)):
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, x, gt=None, flag=False):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        x1 = feat

        to_select['feat'] = x1.squeeze()
        x_class = self.classifier(x1.squeeze())
        to_select['logits'] = x_class
        to_select['probs'] = F.softmax(x_class, dim=1)

        return to_select

    def get_p(self, x):
        for key in self.FC:
            x = self.dropout[key](x)
            x = self.FC[key](x)

        return x

def danet(num_classes, feature_extractor, fx_pretrained=True,
          frozen=[], dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, block_num=5,**kwargs):

    model = DANet(feature_extractor=feature_extractor,
                num_classes=num_classes, frozen=frozen,
                fx_pretrained=fx_pretrained,
                dropout_ratio=dropout_ratio,
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn,
                block_num=block_num, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model