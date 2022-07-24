import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np, h5py
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch
print(torch.__version__)
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import matplotlib.pyplot as plt

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

def plot(L1_avg, psnr_avg, ssim_avg, epoch, save_dir, name):
    plt.plot(np.mean(L1_avg, axis=1)[:epoch])
    plt.savefig(os.path.join(save_dir, name + '_l1_avg_loss.png'))
    plt.close()
    plt.plot(np.mean(psnr_avg, axis=1)[:epoch])
    plt.savefig(os.path.join(save_dir, name + '_mean_psnr.png'))
    plt.close()
    plt.plot(np.std(psnr_avg, axis=1)[:epoch])
    plt.savefig(os.path.join(save_dir, name + '_std_psnr.png'))
    plt.close()
    plt.plot(np.mean(ssim_avg, axis=1)[:epoch])
    plt.savefig(os.path.join(save_dir, name + '_mean_ssim.png'))
    plt.close()

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
        model.set_input(data, one_hot,opt.dataset_name)
        model.optimize_parameters()

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
            visualizer.display_current_results(temp_visuals, epoch, save_result)

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
        model.set_input(data_val, one_hot,opt.dataset_name)
        model.test()

        fake_im=model.fake_B.cpu().data.numpy()
        real_im=model.real_B.cpu().data.numpy()
        real_im=real_im*0.5+0.5
        fake_im=fake_im*0.5+0.5
        fake_im[fake_im<0]=0
        L1_avg[epoch-1,i]=abs(fake_im-real_im).mean()
        psnr_avg[epoch-1,i]=psnr(real_im/real_im.max(), fake_im/fake_im.max())
        ssim_avg[epoch-1,i]=ssim(real_im[0, 0]/real_im.max(), fake_im[0, 0]/fake_im.max())

    if epoch % opt.save_epoch_freq == 0:
        print(dset + ': saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save(dset + '_latest')
        #model.save(epoch)
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

    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()

    #IXI
    #Training data
    opt.dataset_name = "IXI"
    opt.phase='train'
    data_loader_IXI = CreateDataLoader(opt)
    dataset_IXI = data_loader_IXI.load_data()
    dataset_IXI_size = len(data_loader_IXI)
    print('#'+opt.dataset_name+' training images = %d' % dataset_IXI_size)

    #validation data
    opt.phase='val'
    data_loader_val_IXI = CreateDataLoader(opt)
    dataset_val_IXI = data_loader_val_IXI.load_data()
    dataset_size_val = len(data_loader_val_IXI)
    print('#'+opt.dataset_name+' validation images = %d' % dataset_size_val)
    #BRATS
    #Training data
    opt.dataroot = opt.dataroot2
    opt.dataset_name = "BRATS"
    opt.phase='train'
    data_loader_BRATS = CreateDataLoader(opt)
    dataset_BRATS = data_loader_BRATS.load_data()
    dataset_BRATS_size = len(data_loader_BRATS)
    print('#'+opt.dataset_name+' training images = %d' % dataset_BRATS_size)
    #validation data
    opt.phase='val'
    data_loader_val_BRATS = CreateDataLoader(opt)
    dataset_val_BRATS = data_loader_val_BRATS.load_data()
    dataset_size_val = len(data_loader_val_BRATS)
    print('#'+opt.dataset_name+' validation images = %d' % dataset_size_val)
    #MIDAS
    #Training data
    opt.dataroot = opt.dataroot3
    opt.dataset_name = "MIDAS"
    opt.phase='train'
    data_loader_MIDAS = CreateDataLoader(opt)
    dataset_MIDAS = data_loader_MIDAS.load_data()
    dataset_MIDAS_size = len(data_loader_MIDAS)
    print('#'+opt.dataset_name+' training images = %d' % dataset_MIDAS_size)
    #validation data
    opt.phase='val'
    data_loader_val_MIDAS = CreateDataLoader(opt)
    dataset_val_MIDAS = data_loader_val_MIDAS.load_data()
    dataset_size_val = len(data_loader_val_MIDAS)
    print('#'+opt.dataset_name+' validation images = %d' % dataset_size_val)

    #OASIS
    #Training data
    opt.dataroot = opt.dataroot4
    opt.dataset_name = "OASIS"
    opt.phase='train'
    data_loader_OASIS = CreateDataLoader(opt)
    dataset_OASIS = data_loader_OASIS.load_data()
    dataset_OASIS_size = len(data_loader_OASIS)
    print('#'+opt.dataset_name+' training images = %d' % dataset_OASIS_size)
    #validation data
    opt.phase='val'
    data_loader_val_OASIS = CreateDataLoader(opt)
    dataset_val_OASIS = data_loader_val_OASIS.load_data()
    dataset_size_val = len(data_loader_val_OASIS)
    print('#'+opt.dataset_name+' validation images = %d' % dataset_size_val)

    total_size = dataset_IXI_size+dataset_BRATS_size+dataset_MIDAS_size+dataset_OASIS_size
    ixi_coef = dataset_IXI_size/total_size
    brats_coef = dataset_BRATS_size/total_size
    midas_coef = dataset_MIDAS_size/total_size
    oasis_coef = dataset_OASIS_size/total_size
    print(ixi_coef)
    print(brats_coef)
    print(midas_coef)
    print(oasis_coef)



    L1_avg_IXI = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_IXI)])
    psnr_avg_IXI = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_IXI)])
    ssim_avg_IXI = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_IXI)])

    L1_avg_BRATS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_BRATS)])
    psnr_avg_BRATS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_BRATS)])
    ssim_avg_BRATS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_BRATS)])

    L1_avg_MIDAS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MIDAS)])
    psnr_avg_MIDAS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MIDAS)])
    ssim_avg_MIDAS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_MIDAS)])

    L1_avg_OASIS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_OASIS)])
    psnr_avg_OASIS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_OASIS)])
    ssim_avg_OASIS = np.zeros([opt.niter + opt.niter_decay,len(dataset_val_OASIS)])

    model_IXI = create_model(opt)
    visualizer_IXI = Visualizer(opt)
    IXI_netG_SD_9 = model_IXI.netG.model_9.state_dict()
    IXI_netG_SD_10 = model_IXI.netG.model_10.state_dict()
    IXI_netG_SD_11 = model_IXI.netG.model_11.state_dict()
    IXI_netG_SD_12 = model_IXI.netG.model_12.state_dict()
    IXI_netG_SD_13 = model_IXI.netG.model_13.state_dict()
    IXI_netG_SD_14 = model_IXI.netG.model_14.state_dict()
    IXI_netG_SD_15 = model_IXI.netG.model_15.state_dict()
    IXI_mapper = model_IXI.netG.mapper.state_dict()

    model_BRATS = create_model(opt)
    visualizer_BRATS = Visualizer(opt)
    BRATS_netG_SD_9 = model_BRATS.netG.model_9.state_dict()
    BRATS_netG_SD_10 = model_BRATS.netG.model_10.state_dict()
    BRATS_netG_SD_11 = model_BRATS.netG.model_11.state_dict()
    BRATS_netG_SD_12 = model_BRATS.netG.model_12.state_dict()
    BRATS_netG_SD_13 = model_BRATS.netG.model_13.state_dict()
    BRATS_netG_SD_14 = model_BRATS.netG.model_14.state_dict()
    BRATS_netG_SD_15 = model_BRATS.netG.model_15.state_dict()
    BRATS_mapper = model_BRATS.netG.mapper.state_dict()

    model_MIDAS = create_model(opt)
    visualizer_MIDAS = Visualizer(opt)
    MIDAS_netG_SD_9 = model_MIDAS.netG.model_9.state_dict()
    MIDAS_netG_SD_10 = model_MIDAS.netG.model_10.state_dict()
    MIDAS_netG_SD_11 = model_MIDAS.netG.model_11.state_dict()
    MIDAS_netG_SD_12 = model_MIDAS.netG.model_12.state_dict()
    MIDAS_netG_SD_13 = model_MIDAS.netG.model_13.state_dict()
    MIDAS_netG_SD_14 = model_MIDAS.netG.model_14.state_dict()
    MIDAS_netG_SD_15 = model_MIDAS.netG.model_15.state_dict()
    MIDAS_mapper = model_MIDAS.netG.mapper.state_dict()

    model_OASIS = create_model(opt)
    visualizer_OASIS = Visualizer(opt)
    OASIS_netG_SD_9 =  model_OASIS.netG.model_9.state_dict()
    OASIS_netG_SD_10 = model_OASIS.netG.model_10.state_dict()
    OASIS_netG_SD_11 = model_OASIS.netG.model_11.state_dict()
    OASIS_netG_SD_12 = model_OASIS.netG.model_12.state_dict()
    OASIS_netG_SD_13 = model_OASIS.netG.model_13.state_dict()
    OASIS_netG_SD_14 = model_OASIS.netG.model_14.state_dict()
    OASIS_netG_SD_15 = model_OASIS.netG.model_15.state_dict()
    OASIS_mapper = model_OASIS.netG.mapper.state_dict()

    total_steps = 0

    # site information
    one_hot_IXI = torch.tensor([[1.0, 0.0,0.0,0.0]], requires_grad=False ).cuda(opt.gpu_ids[0])
    one_hot_BRATS = torch.tensor([[0.0, 1.0,0.0,0.0]], requires_grad=False).cuda(opt.gpu_ids[0])
    one_hot_MIDAS = torch.tensor([[0.0, 0.0,1.0,0.0]], requires_grad=False ).cuda(opt.gpu_ids[0])
    one_hot_OASIS = torch.tensor([[0.0, 0.0,0.0,1.0]], requires_grad=False ).cuda(opt.gpu_ids[0])

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        #Epoch number
        opt.epoch_number=epoch+1
        #Training step
        opt.dataset_name = 'ixi'
        _,  model_IXI, _, L1_avg_IXI, psnr_avg_IXI, ssim_avg_IXI = one_epoch(opt, dataset_IXI, model_IXI, one_hot_IXI, visualizer_IXI, total_steps,
                                                                           dataset_val_IXI, L1_avg_IXI, psnr_avg_IXI, ssim_avg_IXI, opt.dataset_name)

        plot(L1_avg_IXI, psnr_avg_IXI, ssim_avg_IXI, epoch, save_dir, opt.dataset_name)

        opt.dataset_name = 'brats'
        _,  model_BRATS, _, L1_avg_BRATS, psnr_avg_BRATS, ssim_avg_BRATS = one_epoch(opt, dataset_BRATS, model_BRATS,one_hot_BRATS, visualizer_BRATS, total_steps,
                                                                           dataset_val_BRATS, L1_avg_BRATS, psnr_avg_BRATS, ssim_avg_BRATS, opt.dataset_name)

        plot(L1_avg_BRATS, psnr_avg_BRATS, ssim_avg_BRATS, epoch, save_dir, opt.dataset_name)

        opt.dataset_name = 'midas'
        _,  model_MIDAS, _, L1_avg_MIDAS, psnr_avg_MIDAS, ssim_avg_MIDAS = one_epoch(opt, dataset_MIDAS, model_MIDAS,one_hot_MIDAS, visualizer_MIDAS, total_steps,
                                                                           dataset_val_MIDAS, L1_avg_MIDAS, psnr_avg_MIDAS, ssim_avg_MIDAS, opt.dataset_name)

        plot(L1_avg_MIDAS, psnr_avg_MIDAS, ssim_avg_MIDAS, epoch, save_dir, opt.dataset_name)

        opt.dataset_name = 'oasis'
        _,  model_OASIS, _, L1_avg_OASIS, psnr_avg_OASIS, ssim_avg_OASIS = one_epoch(opt, dataset_OASIS, model_OASIS,one_hot_OASIS, visualizer_OASIS, total_steps,
                                                                           dataset_val_OASIS, L1_avg_OASIS, psnr_avg_OASIS, ssim_avg_OASIS, opt.dataset_name)

        plot(L1_avg_OASIS, psnr_avg_OASIS, ssim_avg_OASIS, epoch, save_dir, opt.dataset_name)

        

        IXI_netG_SD_9 = model_IXI.netG.model_9.state_dict()
        IXI_netG_SD_10 = model_IXI.netG.model_10.state_dict()
        IXI_netG_SD_11 = model_IXI.netG.model_11.state_dict()
        IXI_netG_SD_12 = model_IXI.netG.model_12.state_dict()
        IXI_netG_SD_13 = model_IXI.netG.model_13.state_dict()
        IXI_netG_SD_14 = model_IXI.netG.model_14.state_dict()
        IXI_netG_SD_15 = model_IXI.netG.model_15.state_dict()
        IXI_mapper = model_IXI.netG.mapper.state_dict()


        BRATS_netG_SD_9 = model_BRATS.netG.model_9.state_dict()
        BRATS_netG_SD_10 = model_BRATS.netG.model_10.state_dict()
        BRATS_netG_SD_11 = model_BRATS.netG.model_11.state_dict()
        BRATS_netG_SD_12 = model_BRATS.netG.model_12.state_dict()
        BRATS_netG_SD_13 = model_BRATS.netG.model_13.state_dict()
        BRATS_netG_SD_14 = model_BRATS.netG.model_14.state_dict()
        BRATS_netG_SD_15 = model_BRATS.netG.model_15.state_dict()
        BRATS_mapper = model_BRATS.netG.mapper.state_dict()


        MIDAS_netG_SD_9 = model_MIDAS.netG.model_9.state_dict()
        MIDAS_netG_SD_10 = model_MIDAS.netG.model_10.state_dict()
        MIDAS_netG_SD_11 = model_MIDAS.netG.model_11.state_dict()
        MIDAS_netG_SD_12 = model_MIDAS.netG.model_12.state_dict()
        MIDAS_netG_SD_13 = model_MIDAS.netG.model_13.state_dict()
        MIDAS_netG_SD_14 = model_MIDAS.netG.model_14.state_dict()
        MIDAS_netG_SD_15 = model_MIDAS.netG.model_15.state_dict()
        MIDAS_mapper = model_MIDAS.netG.mapper.state_dict()


        OASIS_netG_SD_9 =  model_OASIS.netG.model_9.state_dict()
        OASIS_netG_SD_10 = model_OASIS.netG.model_10.state_dict()
        OASIS_netG_SD_11 = model_OASIS.netG.model_11.state_dict()
        OASIS_netG_SD_12 = model_OASIS.netG.model_12.state_dict()
        OASIS_netG_SD_13 = model_OASIS.netG.model_13.state_dict()
        OASIS_netG_SD_14 = model_OASIS.netG.model_14.state_dict()
        OASIS_netG_SD_15 = model_OASIS.netG.model_15.state_dict()
        OASIS_mapper = model_OASIS.netG.mapper.state_dict()


        for key in IXI_netG_SD_9:
            IXI_netG_SD_9[key] = (ixi_coef * IXI_netG_SD_9[key] + brats_coef * BRATS_netG_SD_9[key] + midas_coef * MIDAS_netG_SD_9[key] + oasis_coef *OASIS_netG_SD_9[key])

        for key in IXI_netG_SD_10:
            IXI_netG_SD_10[key] = (ixi_coef * IXI_netG_SD_10[key] + brats_coef *BRATS_netG_SD_10[key] + midas_coef *MIDAS_netG_SD_10[key] + oasis_coef *OASIS_netG_SD_10[key])

        for key in IXI_netG_SD_11:
            IXI_netG_SD_11[key] = (ixi_coef * IXI_netG_SD_11[key] + brats_coef *BRATS_netG_SD_11[key] + midas_coef *MIDAS_netG_SD_11[key] + oasis_coef *OASIS_netG_SD_11[key])

        for key in IXI_netG_SD_12:
            IXI_netG_SD_12[key] = (ixi_coef * IXI_netG_SD_12[key] + brats_coef *BRATS_netG_SD_12[key] + midas_coef *MIDAS_netG_SD_12[key] + oasis_coef *OASIS_netG_SD_12[key])

        for key in IXI_netG_SD_13:
            IXI_netG_SD_13[key] = (ixi_coef * IXI_netG_SD_13[key] + brats_coef *BRATS_netG_SD_13[key] + midas_coef *MIDAS_netG_SD_13[key] + oasis_coef *OASIS_netG_SD_13[key])

        for key in IXI_netG_SD_14:
            IXI_netG_SD_14[key] = (ixi_coef * IXI_netG_SD_14[key] + brats_coef *BRATS_netG_SD_14[key] + midas_coef *MIDAS_netG_SD_14[key] + oasis_coef *OASIS_netG_SD_14[key])

        for key in IXI_netG_SD_15:
            IXI_netG_SD_15[key] = (ixi_coef * IXI_netG_SD_15[key] + brats_coef *BRATS_netG_SD_15[key] + midas_coef *MIDAS_netG_SD_15[key] + oasis_coef *OASIS_netG_SD_15[key])

        for key in IXI_mapper:
            IXI_mapper[key] = (ixi_coef * IXI_mapper[key] + brats_coef *BRATS_mapper[key] + midas_coef *MIDAS_mapper[key] + oasis_coef *OASIS_mapper[key])
       

        model_IXI.netG.model_9.load_state_dict(IXI_netG_SD_9)
        model_IXI.netG.model_10.load_state_dict(IXI_netG_SD_10)
        model_IXI.netG.model_11.load_state_dict(IXI_netG_SD_11)
        model_IXI.netG.model_12.load_state_dict(IXI_netG_SD_12)
        model_IXI.netG.model_13.load_state_dict(IXI_netG_SD_13)
        model_IXI.netG.model_14.load_state_dict(IXI_netG_SD_14)
        model_IXI.netG.model_15.load_state_dict(IXI_netG_SD_15)
        model_IXI.netG.mapper.load_state_dict(IXI_mapper)


        model_BRATS.netG.model_9.load_state_dict(IXI_netG_SD_9)
        model_BRATS.netG.model_10.load_state_dict(IXI_netG_SD_10)
        model_BRATS.netG.model_11.load_state_dict(IXI_netG_SD_11)
        model_BRATS.netG.model_12.load_state_dict(IXI_netG_SD_12)
        model_BRATS.netG.model_13.load_state_dict(IXI_netG_SD_13)
        model_BRATS.netG.model_14.load_state_dict(IXI_netG_SD_14)
        model_BRATS.netG.model_15.load_state_dict(IXI_netG_SD_15)
        model_BRATS.netG.mapper.load_state_dict(IXI_mapper)


        model_MIDAS.netG.model_9.load_state_dict(IXI_netG_SD_9)
        model_MIDAS.netG.model_10.load_state_dict(IXI_netG_SD_10)
        model_MIDAS.netG.model_11.load_state_dict(IXI_netG_SD_11)
        model_MIDAS.netG.model_12.load_state_dict(IXI_netG_SD_12)
        model_MIDAS.netG.model_13.load_state_dict(IXI_netG_SD_13)
        model_MIDAS.netG.model_14.load_state_dict(IXI_netG_SD_14)
        model_MIDAS.netG.model_15.load_state_dict(IXI_netG_SD_15)
        model_MIDAS.netG.mapper.load_state_dict(IXI_mapper)


        model_OASIS.netG.model_9.load_state_dict(IXI_netG_SD_9)
        model_OASIS.netG.model_10.load_state_dict(IXI_netG_SD_10)
        model_OASIS.netG.model_11.load_state_dict(IXI_netG_SD_11)
        model_OASIS.netG.model_12.load_state_dict(IXI_netG_SD_12)
        model_OASIS.netG.model_13.load_state_dict(IXI_netG_SD_13)
        model_OASIS.netG.model_14.load_state_dict(IXI_netG_SD_14)
        model_OASIS.netG.model_15.load_state_dict(IXI_netG_SD_15)
        model_OASIS.netG.mapper.load_state_dict(IXI_mapper)



        opt.dataset_name = 'ixi'
        model_IXI.save(opt.dataset_name + '_latest')

        opt.dataset_name = 'brats'
        model_IXI.save(opt.dataset_name + '_latest')

        opt.dataset_name = 'midas'
        model_MIDAS.save(opt.dataset_name + '_latest')

        opt.dataset_name = 'oasis'
        model_OASIS.save(opt.dataset_name + '_latest')
