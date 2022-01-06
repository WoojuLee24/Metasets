from model_utils import *
# from torchmeta.modules.utils import get_subdict
# from torchmeta.modules import (MetaModule, MetaSequential, MetaConv2d, MetaBatchNorm2d, MetaLinear)
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, Linear
import pdb
import os

class Pointnet_cls(Module):
    def __init__(self, num_class=11):
        super(Pointnet_cls, self).__init__()
        self.trans_net1 = transform_net(3, 3)
        self.trans_net2 = transform_net(64, 64)
        self.conv1 = conv_2d(3, 64, 1)
        self.conv2 = conv_2d(64, 64, 1)
        self.conv3 = conv_2d(64, 64, 1)
        self.conv4 = conv_2d(64, 128, 1)
        self.conv5 = conv_2d(128, 1024, 1)

        self.mlp1 = fc_layer(1024, 512)
        self.dropout1 = nn.Dropout2d(p=0.7)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.7)
        self.mlp3 = Linear(256, num_class)

    def forward(self, x, x_point_num=None):
        batch_size = x.size(0)
        point_num = x.size(2)
        
        transform = self.trans_net1(x)
        x = x.transpose(2, 1).contiguous()
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1).contiguous()
        x = self.conv1(x)
        x = self.conv2(x)
        transform = self.trans_net2(x)
        x = x.transpose(2, 1).contiguous()
        x = x.squeeze()
        x = torch.bmm(x, transform)
        x = x.unsqueeze(3)
        x = x.transpose(2, 1).contiguous()
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        if x_point_num is not None:
            '''
            y_i_list = []
            for i in range(batch_size):
                y_i = torch.max(x[i][:,:x_point_num[i]], dim=1, keepdim=False)[0].reshape(1, -1)
                y_i_list.append(y_i)
            x = torch.cat(y_i_list)
            '''
            x = x.squeeze().transpose(-2, -1).contiguous()
            index = torch.arange(point_num).reshape(1, -1).repeat(batch_size, 1).to('cuda')
            mask = (index < x_point_num.reshape(-1, 1)).unsqueeze(-1).repeat(1, 1, 1024)
            x = torch.where(mask, x, torch.zeros(mask.shape).to('cuda')).transpose(-2, -1)
            x, _ = torch.max(x, dim=2, keepdim=False)
        else:
            x, _ = torch.max(x, dim=2, keepdim=False)
            x = x.squeeze()     #batchsize*1024
        x = self.mlp1(x)    #batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)    #batchsize*256
        x = self.dropout2(x)
        x = self.mlp3(x)    #batchsize*10

        return x










