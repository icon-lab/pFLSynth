python3 train_global_aggregation.py \
--gpu_ids 1 \
--dataroot Datasets/IXI/T1_T2__PD/ \
--dataroot2 Datasets/BRATS/T1_T2__FLAIR/ \
--dataroot3 Datasets/MIDAS/T1_T2/ \
--dataroot4 Datasets/OASIS/T1_T2__FLAIR/ \
--name T1_T2_FedGAN_3_heteregeneous_true \
--model federated_synthesis \
--which_model_netG resnet_generator \
--which_direction AtoB \
--lambda_A 100 \
--dataset_mode aligned \
--norm batch \
--n_clients 4 \
--pool_size 0 \
--output_nc 1 \
--input_nc 2 \
--niter 75 \
--niter_decay 75 \
--save_epoch_freq 5 \
--checkpoints_dir checkpoints/ \
--federated_learning 1
