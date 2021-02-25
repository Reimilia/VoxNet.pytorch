import torch
import torch.nn as nn
from collections import OrderedDict
from vit_3d_transformer import ViT3D
from einops import rearrange, repeat

class FeatureTransformer(nn.Module):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(FeatureTransformer, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        if input_shape[0]==32:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                            out_channels=32, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3))
            ]))
        elif input_shape[0]==128:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                            out_channels=32, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3)),
                ('conv3d_3', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu3', torch.nn.ReLU()),
                ('pool3', torch.nn.MaxPool3d(2)),
                ('drop3', torch.nn.Dropout(p=0.3)),
                ('conv3d_4', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu4', torch.nn.ReLU()),
                ('pool4', torch.nn.MaxPool3d(2)),
                ('drop4', torch.nn.Dropout(p=0.3)),
            ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        print(x.shape)
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n
        self.dim_feat = dim_feat

        # One layer transformer
        self.transformer= torch.nn.TransformerEncoderLayer(d_model=x.shape[2]*x.shape[3]*x.shape[4], nhead=4, dropout=0.3)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 256)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(256, self.n_classes))
        ]))

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), x.size(1), -1).transpose(0,1)
        x = self.transformer(x)
        #print(x.shape)
        x = x.transpose(0,1)
        x = x.reshape(x.size(0),-1)
        x = self.mlp(x)
        return x

class NaiveTransformer(nn.Module):
    def __init__(self, n_classes=10, patch_size = 1, feedforward_dim = 1024, mlp_dim = 128, input_shape=(32, 32, 32), num_layers=2, nhead=4):
        super(NaiveTransformer, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.mlp_dim = mlp_dim
        self.patch_size = patch_size

        self.transformer= torch.nn.TransformerEncoderLayer(d_model=patch_size**3, dim_feedforward = feedforward_dim, nhead=nhead, dropout=0.3)
        self.feat = nn.TransformerEncoder(self.transformer, num_layers=num_layers)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], mlp_dim)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(mlp_dim, self.n_classes))
        ]))

        
        
    def forward(self, x):
        p = self.patch_size
        x = rearrange(x, 'b c (h p1) (w p2) (s p3) -> (h w s) b (p1 p2 p3 c)', p1 = p, p2 = p, p3 = p)
        x = self.transformer(x)
        x = x.transpose(0,1)
        x = x.reshape(x.size(0),-1)
        #print(x.shape)
        x = self.mlp(x)
        return x


class TokenizedTransformer(FeatureTransformer):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32)):
        super(FeatureTransformer, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        print(x.shape)
        # (128,128,128)->(30,30,30)
        self.transformer = ViT3D(voxel_size = x.shape[2], dim = 128, patch_size = x.shape[2]//6, depth = 1, heads = 4, mlp_dim = 512, num_classes=n_classes, channels=x.shape[1], dropout=0.3)


    def forward(self, x):
        x = self.feat(x)
        x = self.transformer(x)
        return x

class FPNTransformer(nn.Module):

    def __init__(self, n_classes=10, input_shape=(32, 32, 32), dataset='ModelNet10'):
        super(FPNTransformer, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat1 = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2))
        ]))
        x = self.feat1(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        #print(x.shape)
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        # One layer transformer
        encoded_layer1 = torch.nn.TransformerEncoderLayer(d_model=x.shape[1], nhead=4, dim_feedforward=512, dropout=0.3)
        self.transformer1 = torch.nn.TransformerEncoder(encoded_layer1, num_layers=1)

        self.feat2 = torch.nn.Sequential(OrderedDict([
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3)),
        ]))

        x = self.feat2(x)
        #print(x.shape)
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        # One layer transformer
        encoded_layer2 = torch.nn.TransformerEncoderLayer(d_model=x.shape[1], nhead=4, dim_feedforward=512, dropout=0.3)
        self.transformer2= torch.nn.TransformerEncoder(encoded_layer2, num_layers=1)

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 128)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.4)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))

        pass

    def forward(self, x):
        x = self.feat1(x)
        local_shape = x.shape
        #x = rearrange(x, 'b c h w s -> (h w s) b c')
        #x = self.transformer1(x)
        #x = rearrange(x, '(p1 p2 p3) b c -> b c p1 p2 p3', p1 =local_shape[2], p2=local_shape[3], p3=local_shape[4])
        x= self.feat2(x)
        x = rearrange(x, 'b c h w s -> (h w s) b c')
        x = self.transformer2(x)
        x = rearrange(x, 'f b c -> b (f c)')
        x = self.mlp(x)
        return x