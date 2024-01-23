import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import torch
print(torch.__version__)
import sys
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

def one_epoch(opt, dataset, model,one_hot,  visualizer, total_steps, dataset_val, L1_avg, psnr_avg, ssim_avg, dset):
    opt.phase='train'
    epoch_iter = 0
    iter_data_time = time.time()
    epoch_start_time = time.time()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data, one_hot, dset)
        model.optimize_parameters()
        opt.display_freq = 1
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0

            temp_visuals=model.get_current_visuals()
            if temp_visuals['real_A'].shape[2]==1:
                temp_visuals['real_A']=np.concatenate((temp_visuals['real_A'],np.zeros((temp_visuals['real_A'].shape[0],temp_visuals['real_A'].shape[1],2),dtype=np.uint8)),axis=2)
            elif temp_visuals['real_A'].shape[2]==2:
                temp_visuals['real_A']=np.concatenate((temp_visuals['real_A'],np.zeros((temp_visuals['real_A'].shape[0],temp_visuals['real_A'].shape[1],1),dtype=np.uint8)),axis=2)
            else:
                temp_visuals['real_A']=temp_visuals['real_A'][:,:,0:3]
            if temp_visuals['fake_B'].shape[2]==2:
                temp_visuals['fake_B']=np.concatenate((temp_visuals['fake_B'],np.zeros((temp_visuals['fake_B'].shape[0],temp_visuals['fake_B'].shape[1],1),dtype=np.uint8)),axis=2)
                temp_visuals['real_B']=np.concatenate((temp_visuals['real_B'],np.zeros((temp_visuals['real_B'].shape[0],temp_visuals['real_B'].shape[1],1),dtype=np.uint8)),axis=2)

            temp_visuals['real_B']=temp_visuals['real_B'][:,:,0:3]
            temp_visuals['fake_B']=temp_visuals['fake_B'][:,:,0:3]
            visualizer.display_current_results(temp_visuals, epoch, save_result)#,i)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
        if total_steps % opt.save_latest_freq == 0:
            print(dset + ': saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save(dset + '_latest')

        iter_data_time = time.time()
    #Validation step
    logger = open(os.path.join(save_dir, 'log.txt'), 'a')
    print(opt.dataset_mode)
    opt.phase='val'
    for i, data_val in enumerate(dataset_val):
        model.set_input(data_val, one_hot, dset)
        model.test()

        fake_im=model.fake_B.cpu().data.numpy()
        real_im=model.real_B.cpu().data.numpy()
        real_im=real_im*0.5+0.5
        fake_im=fake_im*0.5+0.5
        fake_im[fake_im<0]=0
        L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
        psnr_avg[epoch-1,i]=psnr(real_im/real_im.max(), fake_im/fake_im.max(), data_range=1)
        ssim_avg[epoch-1,i]=ssim(real_im[0, 0]/real_im.max(), fake_im[0, 0]/fake_im.max(), data_range=1)

    if epoch % opt.save_epoch_freq == 0:
        print(dset + ': saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(dset + '_latest')
        print_log(logger,'Epoch %3d    l1_avg_loss: %.5f    mean_psnr: %.3f    std_psnr:%.3f    mean_ssim: %.3f' % \
        (epoch, np.mean(L1_avg[epoch-1]), np.mean(psnr_avg[epoch-1]), np.std(psnr_avg[epoch-1]), 100 * np.mean(ssim_avg[epoch-1])))
        print_log(logger,'')
        logger.close()

    f = h5py.File(opt.checkpoints_dir+opt.name+'.mat',  "w")
    f.create_dataset(dset + '_L1_avg', data=L1_avg)
    f.create_dataset(dset + '_psnr_avg', data=psnr_avg)
    f.close()
    print(dset + ': End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    model.update_learning_rate()

    return opt, model, total_steps, L1_avg, psnr_avg, ssim_avg


if __name__ == '__main__':
    opt = TrainOptions().parse()
    #Training data
    opt.phase='train'
    opt.dataset_name = "combined"

    data_loader_MRNet = CreateDataLoader(opt)
    dataset_MRNet = data_loader_MRNet.load_data()
    dataset_size = len(data_loader_MRNet)
    print('#'+opt.dataset_name+' training images = %d' % dataset_size)


    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()

    #validation data
    opt.phase='val'

    data_loader_val_MRNet = CreateDataLoader(opt)
    dataset_val_MRNet = data_loader_val_MRNet.load_data()
    dataset_size_val = len(data_loader_val_MRNet)
    print('#'+opt.dataset_name+' validation images = %d' % dataset_size_val)


    L1_avg_MRNet = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MRNet)])
    psnr_avg_MRNet = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MRNet)])
    ssim_avg_MRNet = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MRNet)])

    model_MRNet = create_model(opt)
    visualizer_MRNet = Visualizer(opt)

    MRNet_netG_SD = model_MRNet.netG.state_dict()

    total_steps = 0

    one_hot_MRNet = torch.tensor([[1.0, 0.0,0.0,0.0]], requires_grad=False ).cuda(opt.gpu_ids[0])

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        #Epoch number
        opt.epoch_number=epoch+1
        #Training step
        _,  model_MRNet, _, L1_avg_MRNet, psnr_avg_MRNet, ssim_avg_MRNet = one_epoch(opt, dataset_MRNet, model_MRNet,one_hot_MRNet, visualizer_MRNet, total_steps,
                                                                           dataset_val_MRNet, L1_avg_MRNet, psnr_avg_MRNet, ssim_avg_MRNet, opt.dataset_name)

