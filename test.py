import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.measure import compare_ssim
import torch
def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')

if __name__ == '__main__':

    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)

    if opt.dataset_name == "ixi":
        one_hot = torch.tensor([[1.0, 0.0,0.0,0.0]], requires_grad=False ).cuda(opt.gpu_ids[0])
    elif opt.dataset_name == "brats":
        one_hot = torch.tensor([[0.0, 1.0,0.0,0.0]], requires_grad=False).cuda(opt.gpu_ids[0])
    elif opt.dataset_name == "midas":
        one_hot = torch.tensor([[0.0, 0.0,1.0,0.0]], requires_grad=False ).cuda(opt.gpu_ids[0])
    elif opt.dataset_name == "fastmri":
        one_hot = torch.tensor([[0.0, 0.0,0.0,1.0]], requires_grad=False ).cuda(opt.gpu_ids[0])

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s_%s' % (opt.phase, opt.which_epoch, opt.save_folder_name))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    logger = open(os.path.join(web_dir, 'log.txt'), 'w+')
    print_log(logger, opt.name)
    logger.close()

    # test
    for i, data in enumerate(dataset):
        visualizer.reset()
        model.set_input(data, one_hot)
        model.test()
        fake_im = model.fake_B.cpu().data.numpy()
        real_im = model.real_B.cpu().data.numpy()
        real_im = real_im * 0.5 + 0.5
        fake_im = fake_im * 0.5 + 0.5
        fake_im[fake_im < 0] = 0
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if (i % 100 == 0):
            print('%04d/%d: process image... %s' % (i, len(dataset), img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    webpage.save()

