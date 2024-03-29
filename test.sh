python3 test.py \
--dataroot /auto/data2/odalmaz/Datasets/IXI/T1_T2__PD/ \
--name pFLSynth_variable_setup_exp \
--gpu_ids 0 \
--dataset_mode aligned \
--model federated_synthesis \
--which_model_netG personalized_generator \
--which_direction AtoB \
--norm batch \
--output_nc 1 \
--input_nc 2 \
--checkpoints_dir checkpoints/ \
--phase test \
--how_many 10000 \
--serial_batches \
--results_dir results/ \
--dataset_name ixi \
--save_folder_name IXI \
--n_clients 4 \
--task_name t1_t2
