import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE with minimal modificaiton
    def params(self):
       for name, param in self.named_params(self):
            yield param
    
    def named_leaves(self):
        return []
    
    def named_submodules(self):
        return []
    
    def named_params(self, curr_module=None, memo=None, prefix=''):       
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
                    
        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p
    
    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                if grad is not None:tmp = param_t - lr_inner * grad
                else: tmp = param_t
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self,curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)
            
    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())   
                
    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)

class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
    
    def named_leaves(self):
        if self.bias is not None:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]
    
class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        
        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
    def named_leaves(self):
        if self.bias is not None:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]
    
class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)
        
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size
        self._output_padding = ignore._output_padding
        
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)
        
    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size, self.stride, self.padding, self.kernel_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

    def named_leaves(self):
        if self.bias is not None:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return [('weight', self.weight)]
    
class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)
        
        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:           
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

        
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                        self.training or not self.track_running_stats, self.momentum, self.eps)
            
    def named_leaves(self):
        if self.affine:
            return [('weight', self.weight), ('bias', self.bias)]
        else:
            return []

class LeNet(MetaModule):
    def __init__(self, n_out):
        super(LeNet, self).__init__()
    
        layers = []
        layers.append(MetaConv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))

        layers.append(MetaConv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
        
        layers.append(MetaConv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        
        self.main = nn.Sequential(*layers)
        
        layers = []
        layers.append(MetaLinear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(MetaLinear(84, n_out))
        
        self.fc_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()

class Expression(MetaModule):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)

class Gaussian(MetaModule):
    def __init__(self, mean = 0, var = 1, ratio = 0.05):
        super().__init__()
        self.mean, self.var = mean, var
        self.ratio = ratio

    def forward(self, x):
        if torch.cuda.is_available():
            return x + self.ratio*torch.randn(x.shape).cuda()*self.var**2 + self.mean
        else:
            return x + self.ratio*torch.randn(x.shape)*self.var ** 2 + self.mean


class SVDD(MetaModule):
    def __init__(self, dataset = "mnist"):
        super(SVDD, self).__init__()

        self.rep_dim = 128

        if dataset in ["svhn", "cifar"]:
            self.inputshape = [3,32,32]
        else:
            self.inputshape = [1,32,32]

        enc_layers = []
        # enc_layers.append(Gaussian())
        enc_layers.append(MetaConv2d(self.inputshape[0], 32, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(32, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        enc_layers.append(MetaConv2d(32, 64, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(64, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        enc_layers.append(MetaConv2d(64, 128, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(128, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        self.encoder = nn.Sequential(*enc_layers)

        self.feature_volume = 128 * 4 * 4
        self.z_size = 128
        self.project = MetaLinear(self.feature_volume, self.z_size, bias = False)


    def q(self, x):
        x_ = x.view(-1, self.feature_volume)
        return self.q_mean(x_), self.q_logvar(x_)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if torch.cuda.is_available() else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def forward(self, x):

        encoded = self.encoder(x)

        feats = self.project(encoded.view(-1, self.feature_volume))

        return feats


class AE(MetaModule):
    def __init__(self, dataset="mnist"):
        super(AE, self).__init__()

        self.rep_dim = 128

        if dataset in ["svhn", "cifar"]:
            self.inputshape = [3, 32, 32]
            self.latent_dim = 128
        else:
            self.inputshape = [1, 32, 32]
            self.latent_dim = 32

        enc_layers = []
        # enc_layers.append(Gaussian())
        enc_layers.append(MetaConv2d(self.inputshape[0], self.latent_dim*4, 4, stride = 2, padding=1))
        enc_layers.append(MetaBatchNorm2d(self.latent_dim*4, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())

        enc_layers.append(MetaConv2d(self.latent_dim*4, self.latent_dim*2, 4, stride = 2, padding=1))
        enc_layers.append(MetaBatchNorm2d(self.latent_dim*2, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())

        enc_layers.append(MetaConv2d(self.latent_dim*2, self.latent_dim, 4, stride = 2, padding=1))
        enc_layers.append(MetaBatchNorm2d(self.latent_dim, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []

        dec_layers.append(MetaConvTranspose2d(self.latent_dim, self.latent_dim//2, 4, stride = 2, padding=1))
        dec_layers.append(MetaBatchNorm2d(self.latent_dim//2, eps=1e-04, affine=False))
        dec_layers.append(nn.LeakyReLU())

        dec_layers.append(MetaConvTranspose2d(self.latent_dim//2, self.latent_dim//4, 4, stride = 2, padding=1))
        dec_layers.append(MetaBatchNorm2d(self.latent_dim//4, eps=1e-04, affine=False))
        dec_layers.append(nn.LeakyReLU())

        dec_layers.append(MetaConvTranspose2d(self.latent_dim//4, self.inputshape[0], 4, stride = 2, padding=1))
        dec_layers.append(MetaBatchNorm2d(self.inputshape[0]))
        dec_layers.append(nn.Tanh())

        self.decoder = nn.Sequential(*dec_layers)

        self.feature_volume = 128 * 4 * 4
        self.z_size = 128

        # self.project = MetaLinear(self.feature_volume, self.z_size, bias=False)

    def q(self, x):
        x_ = x.view(-1, self.feature_volume)
        return self.q_mean(x_), self.q_logvar(x_)

    def z(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = (
            Variable(torch.randn(std.size())).cuda() if torch.cuda.is_available() else
            Variable(torch.randn(std.size()))
        )
        return eps.mul(std).add_(mean)

    def forward(self, x):

        encoded = self.encoder(x)

        reconstruct_x = self.decoder(encoded)

        return reconstruct_x

class supervised_model(MetaModule):
    def __init__(self, dataset = "mnist"):
        super(supervised_model, self).__init__()

        self.rep_dim = 128

        if dataset in ["svhn", "cifar"]:
            self.inputshape = [3,32,32]
        else:
            self.inputshape = [1,32,32]

        enc_layers = []
        # enc_layers.append(Gaussian())
        enc_layers.append(MetaConv2d(self.inputshape[0], 32, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(32, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        enc_layers.append(MetaConv2d(32, 64, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(64, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        enc_layers.append(MetaConv2d(64, 128, 5, bias=False, padding=2))
        enc_layers.append(MetaBatchNorm2d(128, eps=1e-04, affine=False))
        enc_layers.append(nn.LeakyReLU())
        enc_layers.append(nn.MaxPool2d(2,2))

        self.encoder = nn.Sequential(*enc_layers)

        self.feature_volume = 128 * 4 * 4
        self.z_size = 128

        self.project = MetaLinear(self.feature_volume, self.z_size, bias = False)

    def forward(self, x):

        encoded = self.encoder(x)

        feats = self.project(encoded.view(-1, self.feature_volume))

        return feats