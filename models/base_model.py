import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, dset,personalized):
        save_filename = '%s_%s_net_%s.pth' % (dset, epoch_label, network_label)
        if personalized:
            save_filename = '%s_personalized_%s_net_%s.pth' % (dset, epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        print(save_path)
        
#        MRNet_state = torch.load("/auto/data2/gelmas/checkpoints/MRNet_usx6/gokber_rGAN/latest_net_G.pth")
#        fastMRI_state = torch.load("/auto/data2/gelmas/checkpoints/fastMRI_usx6/gokber_rGAN/latest_net_G.pth")
#        brats_state = torch.load("/auto/data2/gelmas/checkpoints/brats_usx6/gokber_rGAN/latest_net_G.pth")
#        
#        keys_list = MRNet_state.keys()
#        avg_state = MRNet_state
#        
#        for model_key in keys_list:
#            avg_state[model_key] = ( MRNet_state[model_key] + fastMRI_state[model_key] + brats_state[model_key] ) / 3
        
#        network.load_state_dict(avg_state)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
