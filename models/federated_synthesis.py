import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np



class federated_synthesis(BaseModel):
    def name(self):
        return 'federated_synthesis'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(1, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, opt.init_type,
                                      self.gpu_ids,opt.n_clients,opt.mapping_layers)


        self.task = opt.task_name
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_1 = networks.define_D(1 + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,opt.output_nc)
            self.netD_2 = networks.define_D(1 + opt.output_nc, opt.ndf,
                                              opt.which_model_netD,
                                              opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,opt.output_nc)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch, opt.dataset_name,opt.personalized)
            if self.isTrain:
                self.load_network(self.netD_1, 'D_1', opt.which_epoch, opt.dataset_name)
                self.load_network(self.netD_2, 'D_2', opt.which_epoch, opt.dataset_name)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_1 = torch.optim.Adam(self.netD_1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_2 = torch.optim.Adam(self.netD_2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_1)
            self.optimizers.append(self.optimizer_D_2)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD_1)
        print('-----------------------------------------------')

    def set_input_test(self, input, latent,dataset_name):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        if dataset_name == 'ixi':
            if self.task == 't1_t2':
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
            else:
                #T2->PD
                self.input_A = torch.unsqueeze(input_A[:,1],axis=1)
                self.input_B = input_B 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 1.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])

        if dataset_name == 'brats':
            if self.task == 't1_t2':
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,    0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
            else:
                #FLAIR->T2
                self.input_A =  input_B 
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1) 
                self.task_info = torch.tensor([[0.0, 0.0 , 0.0,  1.0,   0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])

        if dataset_name == 'midas':
            if self.task == 't1_t2':
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_B[:,0],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,     0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
            else:
                #T2->T1
                self.input_A = torch.unsqueeze(input_B[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,0],axis=1) 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])

        if dataset_name == 'fastmri':
            if self.task == 't1_t2':
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,     0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
            else:
                #T2->FLAIR
                self.input_A = torch.unsqueeze(input_A[:,1],axis=1)
                self.input_B = input_B 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,     0.0, 0.0, 0.0, 1.0]], requires_grad=False ).cuda(self.gpu_ids[0])

        self.latent = latent
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def set_input(self, input, latent,dataset_name):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']

        if len(self.gpu_ids) > 0:

            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)

        if dataset_name == 'ixi':
            if np.random.uniform() >= 0.5:
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 1
            else:
                #T2->PD
                self.input_A = torch.unsqueeze(input_A[:,1],axis=1)
                self.input_B = input_B 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,0.0, 0.0, 1.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 0
        if dataset_name == 'brats':
            if np.random.uniform() >= 0.5:
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 1
            else:
                #FLAIR->T2
                self.input_A =  input_B 
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1) 
                self.task_info = torch.tensor([[0.0, 0.0 , 0.0,  1.0,0.0, 1.0, 0.0, 0.0 ]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 0

        if dataset_name == 'midas':
            if np.random.uniform() >= 0.5:
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_B[:,0],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 1
            else:
                #T2->T1
                self.input_A = torch.unsqueeze(input_B[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,0],axis=1) 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,   1.0, 0.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 0

        if dataset_name == 'fastmri':
            if np.random.uniform() >= 0.5:
                #T1->T2
                self.input_A = torch.unsqueeze(input_A[:,0],axis=1)
                self.input_B = torch.unsqueeze(input_A[:,1],axis=1)
                self.task_info = torch.tensor([[1.0, 0.0, 0.0, 0.0,   0.0, 1.0, 0.0, 0.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 1
            else:
                #T2->FLAIR
                self.input_A = torch.unsqueeze(input_A[:,1],axis=1)
                self.input_B = input_B 
                self.task_info = torch.tensor([[0.0, 1.0, 0.0, 0.0,   0.0, 0.0, 0.0, 1.0]], requires_grad=False ).cuda(self.gpu_ids[0])
                self.which_D = 0

        self.latent = latent
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.latent = torch.cat([self.latent,self.task_info],axis=1)
        self.fake_B = self.netG(self.real_A,self.latent)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.latent = torch.cat([self.latent,self.task_info],axis=1)
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG(self.real_A,self.latent)
            self.real_B = Variable(self.input_B)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        if self.which_D:
            pred_fake = self.netD_1(fake_AB.detach())
        else:
            pred_fake = self.netD_2(fake_AB.detach())

        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        
        if self.which_D:
            pred_real = self.netD_1(real_AB)
        else:
            pred_real = self.netD_2(real_AB)

        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_adv

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)


        if self.which_D:
            pred_fake = self.netD_1(fake_AB)
        else:
            pred_fake = self.netD_2(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_adv
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * 1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D_1.zero_grad()
        self.optimizer_D_2.zero_grad()
        self.backward_D()
        self.optimizer_D_1.step()
        self.optimizer_D_2.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data),
                            ('G_L1', self.loss_G_L1.data),
                            ('D_real', self.loss_D_real.data),
                            ('D_fake', self.loss_D_fake.data)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD_1, 'D_1', label, self.gpu_ids)
        self.save_network(self.netD_2, 'D_2', label, self.gpu_ids)

