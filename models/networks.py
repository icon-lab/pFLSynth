import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal',
             gpu_ids=[],n_clients=2,mapping_layers=2):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())

    if which_model_netG == 'personalized_generator':
        netG = personalized_generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                gpu_ids=gpu_ids,n_clients=n_clients,mapping_layers = mapping_layers,n_contrasts=4)
    
    elif which_model_netG == 'unet_fedmm':
        netG =Unet_mmGAN(in_chans=2,out_chans=2)

    elif which_model_netG == 'resnet_generator':
        netG = resnet_generator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                                gpu_ids=gpu_ids)

    elif which_model_netG == 'unet_generator':
        netG = Unet()

    elif which_model_netG == 'switchable_unet':
        netG = switchable_Unet()

    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    if not which_model_netG == 'unet_fedmm' and not which_model_netG == 'unet_generator' and not which_model_netG == 'switchable_unet':
        init_weights(netG, init_type=init_type)
    return netG






def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], output_nc=1):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(gpu_ids[0])
    init_weights(netD, init_type=init_type)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class Mapper(nn.Module):

    def __init__(self, latent_size=2, dlatent_size=512, dlatent_broadcast=None,
                 mapping_layers=8, mapping_fmaps=512, mapping_lrmul=0.01, mapping_nonlinearity='lrelu',
                 use_wscale=True, normalize_latents=True, **kwargs):
        """
        Mapping network used in the StyleGAN paper.
        :param latent_size: Latent vector(Z) dimensionality.
        # :param label_size: Label dimensionality, 0 if no labels.
        :param dlatent_size: Disentangled latent (W) dimensionality.
        :param dlatent_broadcast: Output disentangled latent (W) as [minibatch, dlatent_size]
                                  or [minibatch, dlatent_broadcast, dlatent_size].
        :param mapping_layers: Number of mapping layers.
        :param mapping_fmaps: Number of activations in the mapping layers.
        :param mapping_lrmul: Learning rate multiplier for the mapping layers.
        :param mapping_nonlinearity: Activation function: 'relu', 'lrelu'.
        :param use_wscale: Enable equalized learning rate?
        :param normalize_latents: Normalize latent vectors (Z) before feeding them to the mapping layers?
        :param kwargs: Ignore unrecognized keyword args.
        """

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # # Normalize latents.
        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)

        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x
class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class AdaCW(nn.Module):
    def __init__(self, channel=256, reduction=8,latent_size=512):
        super(AdaCW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(latent_size, channel // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x,latent):
        b, c, _, _ = x.size()
        latent = self.fc(latent).view(b, c, 1, 1)
        return x * latent.expand_as(x)

class personalization_block(nn.Module):
    def __init__(self, channels=256, reduction=8,dlatent_size=512, use_instance_norm=True,use_styles=True):
        super(personalization_block, self).__init__()
        self.adain = AdaIN(channels=channels, dlatent_size=dlatent_size, use_instance_norm=use_instance_norm,use_styles=use_styles)
        self.adacw = AdaCW(channel = channels,reduction=reduction, latent_size=dlatent_size)
    def forward(self, x, latent):
        out = self.adain(x,latent)
        out = self.adacw(out,latent)
        return out

class personalized_generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect',n_clients=4,mapping_layers = 2,n_contrasts=4):
        assert (n_blocks >= 0)
        super(personalized_generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_clients = n_clients
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ############################################################################################
        # Layer1-Encoder1
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        setattr(self, 'model_1', nn.Sequential(*model))
        ############################################################################################
        # Layer2-Encoder2
        n_downsampling = 2
        model = []
        i = 0
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'model_2', nn.Sequential(*model))
        ############################################################################################
        # Layer3-Encoder3
        model = []
        i = 1
        mult = 2 ** i
        model = [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                           stride=2, padding=1, bias=use_bias),
                 norm_layer(ngf * mult * 2),
                 nn.ReLU(True)]
        setattr(self, 'model_3', nn.Sequential(*model))

        ############################################################################################
        # Mapper
        self.mapper = Mapper(latent_size=n_clients+2*n_contrasts, dlatent_size=512, mapping_layers=mapping_layers)

        #Personalization Blocks
        self.personalization_blocks = nn.ModuleList([])
        
        self.personalization_blocks.append( personalization_block(channels=64, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=128, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                    use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=256, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=128, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))
        self.personalization_blocks.append( personalization_block(channels=64, dlatent_size=512, use_instance_norm=True,
                                     use_styles=True))


        mult = 4
        self.model_4 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_5 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_6 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_7 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_8 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_9 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_10 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_11 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        self.model_12 = residual_block(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        
       
        # Layer13-Decoder1
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'model_13', nn.Sequential(*model))
        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias),
                 norm_layer(int(ngf * mult / 2)),
                 nn.ReLU(True)]
        setattr(self, 'model_14', nn.Sequential(*model))
        # Layer15-Decoder3
        model = []
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, 'model_15', nn.Sequential(*model))

    ############################################################################################

    def forward(self, input, site_task_info):
        # produce latents via site and task info
        latent = self.mapper(site_task_info)
        # encoder
        x1 = self.model_1(input)
        x1 = self.personalization_blocks[0](x1, latent)
        x2 = self.model_2(x1)
        x2 = self.personalization_blocks[1](x2, latent)
        x3 = self.model_3(x2)
        x3 = self.personalization_blocks[2](x3, latent)
        #residual bottleneck
        x4 = self.model_4(x3)
        x4 = self.personalization_blocks[3](x4, latent)
        x5 = self.model_5(x4)
        x5 = self.personalization_blocks[4](x5, latent)
        x6 = self.model_6(x5)
        x6 = self.personalization_blocks[5](x6, latent)
        x7 = self.model_7(x6)
        x7 = self.personalization_blocks[6](x7, latent)
        x8 = self.model_8(x7)
        x8 = self.personalization_blocks[7](x8, latent)
        x9 = self.model_9(x8)
        x9 = self.personalization_blocks[8](x9, latent)
        x10 = self.model_10(x9)
        x10 = self.personalization_blocks[9](x10, latent)
        x11 = self.model_11(x10)
        x11 = self.personalization_blocks[10](x11, latent)
        x = self.model_12(x11)
        x = self.personalization_blocks[11](x, latent)
        # decoder
        x = self.model_13(x)
        x = self.personalization_blocks[12](x, latent)
        x = self.model_14(x)
        x = self.personalization_blocks[13](x, latent)
        x = self.model_15(x)
        return x

class residual_block(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(residual_block, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = x + out
        return out

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(self, input_size, output_size, gain=2 ** 0.5, use_wscale=False, lrmul=1, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x


class AdaIN(nn.Module):

    def __init__(self, channels, dlatent_size, use_instance_norm, use_styles):
        super().__init__()

        layers = []
        if use_instance_norm:
            layers.append(('instance_norm', nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=True)
        else:
            self.style_mod = None

    def forward(self, x, dlatents_in_slice=None):
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class ShiftedRelu(nn.Module):

    def __init__(self):
        super(ShiftedRelu, self).__init__()
        self.shift = nn.Parameter(torch.ones(1))

    def forward(self, input):
        return torch.max(input, -self.shift)


class ShiftLayer(nn.Module):

    def __init__(self):
        super(ShiftLayer, self).__init__()
        self.shift = nn.Parameter(torch.ones(1))

    def forward(self, input):
        return input - self.shift



class Unet_mmGAN(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int=2,
        out_chans: int=2,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tanh = nn.Tanh()
        # self.first_layer = ConvBlock(in_chans, chans, drop_prob)
        self.down_sample_layers =nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        
        )


    def forward(self, image: torch.Tensor, latent,direction) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        output = image

        if direction:
            output = torch.cat([output, output], dim=1)
            output[:,1] = -1
        else:
            output = torch.cat([output, output], dim=1)
            output[:,0] = -1
        stack = []
        
        # output = self.first_layer(output)
        # stack.append(output)
        #first layer


        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        output = self.tanh(output)

        if direction:
            output = output[:,1]
        else:
            output = output[:,0]
        output = torch.unsqueeze(output, 1)
        return output


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
           nn.InstanceNorm2d(out_chans),
           # nn.BatchNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            print(self.model(input).size())
            return self.model(input)


class resnet_generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 gpu_ids=[], padding_type='reflect', down_samp=1, fusion_layer_level=1):
        assert (n_blocks >= 0)
        super(resnet_generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.down_samp = down_samp
        self.fusion_layer_level = fusion_layer_level
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        ############################################################################################
        # Layer1-Encoder1
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,bias=use_bias)
        self.norm1_1 = norm_layer(ngf)
        self.norm1_2 = norm_layer(ngf)
        self.relu = nn.ReLU(True)
        n_downsampling = 2
        i = 0
        mult = 2 ** i
        self.conv2 = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,stride=2, padding=1, bias=use_bias)
        self.norm2_1 = norm_layer(ngf * mult * 2)
        self.norm2_2 = norm_layer(ngf * mult * 2)
        i = 1
        mult = 2 ** i
        self.conv3 = nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,stride=2, padding=1, bias=use_bias)
        self.norm3_1 = norm_layer(ngf * mult * 2)
        self.norm3_2 = norm_layer(ngf * mult * 2)

        mult = 2 ** n_downsampling
        self.model_4 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_5 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_6 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_7 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_8 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_9 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_10 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_11 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
        self.model_12 = ResnetBlock_two_norms(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=True,use_bias=use_bias)
    

    # ############################################################################################
        i = 0
        mult = 2 ** (n_downsampling - i)
        self.conv13 = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias)
        self.norm13_1 = norm_layer(int(ngf * mult / 2))
        self.norm13_2 = norm_layer(int(ngf * mult / 2))
        i = 1
        mult = 2 ** (n_downsampling - i)
        self.conv14 = nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=2,
                                    padding=1, output_padding=1,
                                    bias=use_bias)
        self.norm14_1 = norm_layer(int(ngf * mult / 2))
        self.norm14_2 = norm_layer(int(ngf * mult / 2))

        self.pad15 = nn.ReflectionPad2d(3)
        self.conv15 = nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0,bias=use_bias)
        self.bias15_1 = torch.Tensor(1, 256, 256)
        self.bias15_1 = nn.Parameter(nn.init.xavier_uniform_(self.bias15_1))
        self.bias15_2 = torch.Tensor(1, 256, 256)
        self.bias15_2 = nn.Parameter(nn.init.xavier_uniform_(self.bias15_1))
        self.tanh15 = nn.Tanh()


    def forward(self, input,direction=True,latent =True):

        direction=True
        #encoder
        #model_1
        x = self.pad1(input) 
        x = self.conv1(x)
        if direction:
           x = self.norm1_1(x)
        else:
           x = self.norm1_2(x)
        x = self.relu(x)
        #model_2
        x = self.conv2(x)
        if direction:
           x = self.norm2_1(x)
        else:
           x = self.norm2_2(x)
        x = self.relu(x)
        #model_3
        x = self.conv3(x)
        if direction:
           x = self.norm3_1(x)
        else:
           x = self.norm3_2(x)
        x = self.relu(x)
        #residual blocks

        x = self.model_4(x,direction)
        x = self.model_5(x,direction)
        x = self.model_6(x,direction)
        x = self.model_7(x,direction)
        x = self.model_8(x,direction)
        x = self.model_9(x,direction)
        x = self.model_10(x,direction)
        x = self.model_11(x,direction)
        x = self.model_12(x,direction)
        #model_13
        x = self.conv13(x)
        if direction:
           x = self.norm13_1(x)
        else:
           x = self.norm13_2(x)
        x = self.relu(x)
        #model_14
        x = self.conv14(x)
        if direction:
           x = self.norm14_1(x)
        else:
           x = self.norm14_2(x)
        x = self.relu(x)
        #model_15
        x = self.pad15(x)
        x = self.conv15(x)
        if direction:
           x = x + self.bias15_1
        else:
           x = x + self.bias15_2
        x = self.tanh15(x)
        return x
class ResnetBlock_two_norms(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock_two_norms, self).__init__()
        p = 0
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)
        self.bn1_1 = norm_layer(dim)
        self.bn1_2 = norm_layer(dim)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.pad_2 = nn.ReflectionPad2d(1)
        self.conv_2 = nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias)
        self.bn2_1 = norm_layer(dim)
        self.bn2_2 = norm_layer(dim)
    
    def forward(self, x, direction):
        out = self.pad_1(x)
        out = self.conv_1(out)
        if direction:
            out=self.bn1_1(out)
        else:
            out=self.bn1_2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.pad_2(out)
        out= self.conv_2(out)
        if direction:
            out=self.bn2_1(out)
        else:
            out=self.bn2_2(out)
        out = x + out
        return out

class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int=1,
        out_chans: int=1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tanh = nn.Tanh()
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        
        )


    def forward(self, image: torch.Tensor, latent) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        output = self.tanh(output)
        return output

class switchable_Unet(nn.Module):
    """
    PyTorch implementation of a Switchable U-Net model.
    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int=1,
        out_chans: int=1,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        mapping_layers: int = 2,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tanh = nn.Tanh()
        # mapper
        #mapper only knows the direction of translation
        self.mapper = Mapper(latent_size=2, dlatent_size=512, mapping_layers=mapping_layers)

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.down_AdaIn_layers = nn.ModuleList([AdaIN(channels=chans, dlatent_size=512, use_instance_norm=True,use_styles=True)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            self.down_AdaIn_layers.append(AdaIN(channels=ch * 2, dlatent_size=512, use_instance_norm=True,use_styles=True))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob)
        self.conv_AdaIn = AdaIN(channels=ch * 2, dlatent_size=512, use_instance_norm=True,use_styles=True)
        self.up_conv = nn.ModuleList()
        self.up_conv_AdaIn = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        self.up_transpose_conv_AdaIn = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_transpose_conv_AdaIn.append(AdaIN(channels=ch, dlatent_size=512, use_instance_norm=True,use_styles=True))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_conv_AdaIn.append(AdaIN(channels=ch, dlatent_size=512, use_instance_norm=True,use_styles=True))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_transpose_conv_AdaIn.append(AdaIN(channels=ch, dlatent_size=512, use_instance_norm=True,use_styles=True))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        
        
        )
        self.up_conv_AdaIn.append(AdaIN(channels=self.out_chans, dlatent_size=512, use_instance_norm=True,use_styles=True))

    def forward(self, image: torch.Tensor, task_info) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image
        latent = self.mapper(task_info)
        # apply down-sampling layers
        for layer,adain in zip(self.down_sample_layers,self.down_AdaIn_layers):
            output = layer(output)
            output = adain(output,latent)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)
        output = self.conv_AdaIn(output,latent)
        # print(output.size())
        # apply up-sampling layers
        for transpose_conv, transpose_conv_adain, conv, conv_adain in zip(self.up_transpose_conv,self.up_transpose_conv_AdaIn, self.up_conv,self.up_conv_AdaIn):
            
            downsample_layer = stack.pop()
            output = transpose_conv(output)
            output = transpose_conv_adain(output,latent)


            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
            output = conv_adain(output,latent)
            
            

        output = self.tanh(output)
        return output