import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
import math
import sys
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode =='fastmri':
        target_file='/auto/data2/korkmaz/DIP_PROJECT/fastmri/brain/'+opt.output_mod+'/'+opt.output_mod+'_under_sampled_'+str(opt.us_t1)+'x_multicoil_'+str(opt.phase)+'.mat'
        f = h5py.File(target_file,'r') 
        data_us_t1=np.transpose(np.array(f['images_us']),(0,1,3,2))
        data_us_t1=data_us_t1['real']+ 1j *data_us_t1['imag']        
        coil_maps=np.transpose(np.array(f['coil_maps']),(0,1,3,2))
        coil_maps=coil_maps['real']+ 1j *coil_maps['imag'] 
        data_us_t1=np.sum(np.conjugate(coil_maps)*data_us_t1,axis=0)
        data_us_t1=np.expand_dims(data_us_t1,0)
        data_us_t1_mag=np.absolute(data_us_t1)
        data_us_t1_phase=np.angle(data_us_t1)/(2*np.pi)+1

        data_fs_t1=np.transpose(np.array(f['images_fs']),(0,2,1))
        data_fs_t1=data_fs_t1['real']+ 1j *data_fs_t1['imag']  
        data_fs_t1=np.expand_dims(data_fs_t1,0)
        data_fs=np.absolute(data_fs_t1)   
        
#        
        
        subjects=data_fs.shape[1]/10
        print(subjects)    
        data_us_t1_mag=np.split(data_us_t1_mag,subjects,axis=1)
        norm_t1=[x.max() for x in data_us_t1_mag]
        data_us_t1_mag=[x/x.max() for x in data_us_t1_mag]            
        data_us_t1_mag=np.concatenate(data_us_t1_mag,axis=1)
        data_us_t1=np.concatenate((data_us_t1_mag,data_us_t1_phase),axis=0)

        
        data_fs=np.split(data_fs,subjects,axis=1)
        for ii in range(len(data_fs)):
            data_fs[ii]=data_fs[ii]/norm_t1[ii]

        data_fs=np.concatenate(data_fs,axis=1)
        print(data_fs.max())
        data_us=data_us_t1        

        dataset=[]                       
        for train_sample in range(data_us.shape[1]):
            data_us[:,train_sample,:,:]=(data_us[:,train_sample,:,:]-0.5)/0.5
            data_fs[:,train_sample,:,:]=(data_fs[:,train_sample,:,:]-0.5)/0.5
            dataset.append({'A': torch.from_numpy(data_us[:,train_sample,:,:]), 'B':torch.from_numpy(data_fs[:,train_sample,:,:]), 
            'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
        print('#training images = %d' % train_sample)
        print(data_us.shape)
        print(data_fs.shape)  

#    elif opt.dataset_mode =='gokberk':
#        #target_file='/auto/data2/salman/datasets/recsynth/IXI_gaussian/T1_'+str(opt.us_t1)+'_multi_synth_recon_'+str(opt.phase)+'.mat'
#	#target_file = '/auto/data2/gelmas/new_fastMRI/'+str(opt.phase)+'/data.mat'
#	target_file = '/auto/data2/gelmas/fastMRI_usx6/'+str(opt.phase)+'/data.mat'       
## target_file = '/auto/data2/gelmas/recon_rGAN_MRNet-v1.0/'+str(opt.phase)+'/data.mat'
#	#target_file = '/auto/data2/gelmas/recon_rGAN_fastMRI_knee/'+str(opt.phase)+'/data.mat' 
#	#target_file = '/auto/data2/gelmas/recon_rGAN_brats/'+str(opt.phase)+'/data.mat' 
#
#
#        #target_file='/auto/data2/korkmaz/DIP_PROJECT/fastmri/brain/'+opt.output_mod+'/'+opt.output_mod+'_under_sampled_'+str(opt.us_t1)+'x_multicoil_'+str(opt.phase)+'.mat'
#        f = h5py.File(target_file,'r') 
#        data_us=np.transpose(np.array(f['data_us']),(0,1,3,2))
#        data_us=data_us.astype(np.float32)
#        #data_fs=np.transpose(np.array(f['images_fs']),(0,1,3,2))
#        #data_fs=data_fs.astype(np.float32)
#
#        data_fs=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
#        data_fs=data_fs.astype(np.float32)      
#        dataset=[]                       
#        for train_sample in range(data_us.shape[1]):
#            data_us[:,train_sample,:,:]=(data_us[:,train_sample,:,:]-0.5)/0.5
#            data_fs[:,train_sample,:,:]=(data_fs[:,train_sample,:,:]-0.5)/0.5
#            dataset.append({'A': torch.from_numpy(data_us[:,train_sample,:,:]), 'B':torch.from_numpy(data_fs[:,train_sample,:,:]), 
#            'A_paths':opt.dataroot, 'B_paths':opt.dataroot})
#        print('#training images = %d' % train_sample)
#        print(data_us.shape)
#        print(data_fs.shape)
    elif opt.dataset_mode =='aligned':
        if opt.dataset_mode == 'aligned':
            from data.aligned_dataset import AlignedDataset
            dataset = AlignedDataset()
        elif opt.dataset_mode == 'unaligned':
            from data.unaligned_dataset import UnalignedDataset
            dataset = UnalignedDataset()
        elif opt.dataset_mode == 'single':
            from data.single_dataset import SingleDataset
            dataset = SingleDataset()
        else:
            raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)  
            print("dataset [%s] was created" % (dataset.name()))
        dataset.initialize(opt)
    elif opt.dataset_mode =='gokberk':
        if str(opt.phase) != "test":
            # if str(opt.phase) == "train":
            #     n = 420
            # elif str(opt.phase) == "val":
            #     n = 53
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/'+opt.dataset_name+'_usx6/'+str(opt.phase)+'/data.mat'
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/poisson/IXI_usx6/'+str(opt.phase)+'/data.mat'
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/IXI_usx'+opt.acc_rate+'/'+'train/data.mat'
            print(target_file)
            print("hello 31012022")
            
            f = h5py.File(target_file,'r') 
            data_us_IXI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            # data_us_IXI=data_us_IXI.astype(np.float32)
            us_mask_IXI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            # us_mask_IXI=us_mask_IXI.astype(np.float32)  
            data_fs_IXI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            # data_fs_IXI=data_fs_IXI.astype(np.float32) 
            
            print("us real ",data_us_IXI.shape)
            print("fs ",data_fs_IXI.shape)
            print("mask ",us_mask_IXI.shape)
            print("hello 31012022")
            
            
            
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/poisson/fastMRI_usx6/'+str(opt.phase)+'/data.mat'
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/fastMRI_usx'+opt.acc_rate+'/'+'train/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_fastMRI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            # data_us_fastMRI=data_us_fastMRI.astype(np.float32)
            us_mask_fastMRI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            # us_mask_fastMRI=us_mask_fastMRI.astype(np.float32)  
            data_fs_fastMRI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            # data_fs_fastMRI=data_fs_fastMRI.astype(np.float32) 
            
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/poisson/brats_usx6/'+str(opt.phase)+'/data.mat'
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/brats_usx'+opt.acc_rate+'/'+'train/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_brats=np.transpose(np.array(f['data_us']),(0,1,3,2))
            # data_us_brats=data_us_brats.astype(np.float32)
            us_mask_brats=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            # us_mask_brats=us_mask_brats.astype(np.float32)  
            data_fs_brats=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            # data_fs_brats=data_fs_brats.astype(np.float32) 
            # return 0
            data_us = np.concatenate((data_us_IXI.astype(np.float32) ,data_us_fastMRI.astype(np.float32) ,data_us_brats.astype(np.float32) ),axis=1 )
            us_mask = np.concatenate((us_mask_IXI.astype(np.float32) ,us_mask_fastMRI.astype(np.float32) ,us_mask_brats.astype(np.float32) ),axis=1 )
            data_fs = np.concatenate((data_fs_IXI.astype(np.float32) ,data_fs_fastMRI.astype(np.float32) ,data_fs_brats.astype(np.float32) ),axis=1 )
            
            for i in range(us_mask.shape[1]):
                if np.array_equal(data_us[0,i,:,:], np.zeros([256,256]).astype(np.float32)) or np.array_equal(data_us[1,i,:,:], np.zeros([256,256]).astype(np.float32)):
                    print("error in ", i)
                if np.array_equal(data_fs[0,i,:,:], np.zeros([256,256]).astype(np.float32)) or np.array_equal(us_mask[0,i,:,:], np.zeros([256,256]).astype(np.float32)):
                    print("error in ", i)
                
            
            
            shuff_vec = np.arange(opt.slice_num)
            np.random.shuffle(shuff_vec)
            data_us = data_us[:,shuff_vec,:,:]
            data_fs = data_fs[:,shuff_vec,:,:]
            us_mask = us_mask[:,shuff_vec,:,:]
            
            # print(np.array_equal(np.concatenate((data_us_IXI,data_us_fastMRI,data_us_brats),axis=1 ),data_us))
            
        else:
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/'+opt.dataset_name+'_usx6/'+str(opt.phase)+'/data.mat'
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+"/"+opt.dataset_name+'_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/fake_420_slices/'+opt.us_pattern+"/"+opt.dataset_name+'_usx'+opt.acc_rate+'/'+"val"+'/data.mat'
            print(target_file)
            #target_file = '/auto/data2/gelmas/recon_rGAN_fastMRI_knee/'+str(opt.phase)+'/data.mat' 
     	#target_file = '/auto/data2/gelmas/recon_rGAN_brats/'+str(opt.phase)+'/data.mat' 
    
    
            #target_file='/auto/data2/korkmaz/DIP_PROJECT/fastmri/brain/'+opt.output_mod+'/'+opt.output_mod+'_under_sampled_'+str(opt.us_t1)+'x_multicoil_'+str(opt.phase)+'.mat'
            f = h5py.File(target_file,'r') 
            data_us=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us=data_us.astype(np.float32)
            us_mask=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask=us_mask.astype(np.float32)  
            #data_fs=np.transpose(np.array(f['images_fs']),(0,1,3,2))
            #data_fs=data_fs.astype(np.float32)
    
            data_fs=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs=data_fs.astype(np.float32)  
            
        print(data_us.shape)
        print(us_mask.shape)
        print(data_fs.shape)  
        # ind = 0
        # print(np.amax( np.sqrt(np.square(data_us[0,ind,:,:]) + np.square(data_us[1,ind,:,:]) ) ), " ", np.amin( np.sqrt(np.square(data_us[0,ind,:,:]) + np.square(data_us[1,ind,:,:]) ) ))
        # print(np.amax( data_fs[0,ind,:,:]), " ", np.amin( data_fs[0,ind,:,:]))
        # print(np.amax( us_mask[0,ind,:,:]), " ", np.amin( us_mask[0,ind,:,:]))
        # return 0
        dataset=[]                       
        # for train_sample in range(data_us.shape[1]):
        for train_sample in range(opt.slice_num):
        # for train_sample in range(357):
            # print("US BEFORE MIN: ", np.min(data_us[:,train_sample,:,:]))
            # print("US BEFORE MAX: ", np.max(data_us[:,train_sample,:,:]))
            data_us[:,train_sample,:,:]=(data_us[:,train_sample,:,:]-0.5)/0.5
            # print("US AFTER MIN: ", np.min(data_us[:,train_sample,:,:]))
            # print("US AFTER MAX: ", np.max(data_us[:,train_sample,:,:]))
            temp = us_mask[:,train_sample,:,:].reshape(256,256)
 	    #print(us_mask.shape)
 	    #print(asd)	
            # print("us: ",np.amax( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) ), np.amin( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) ), " fs: ",np.amax( data_fs[0,train_sample,:,:]), np.amin( data_fs[0,train_sample,:,:]))
            # if np.isnan(np.amax( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) )):
            #     print(train_sample)
                
            temp = np.fft.ifftshift(temp)
            us_mask[:,train_sample,:,:]=temp
            # print("FS BEFORE MIN: ", np.min(data_fs[:,train_sample,:,:]))
            # print("FS BEFORE MAX: ", np.max(data_fs[:,train_sample,:,:]))
            data_fs[:,train_sample,:,:]=(data_fs[:,train_sample,:,:]-0.5)/0.5
            # print("FS AFTER MIN: ", np.min(data_fs[:,train_sample,:,:]))
            # print("FS AFTER MAX: ", np.max(data_fs[:,train_sample,:,:]))
            dataset.append({'A': torch.from_numpy(data_us[:,train_sample,:,:]), 'B':torch.from_numpy(data_fs[:,train_sample,:,:]),
                      'C': torch.from_numpy(us_mask[:,train_sample,:,:]),      
            'A_paths':opt.dataroot, 'B_paths':opt.dataroot, 'C_paths':opt.dataroot})
        # return 0
        print('#training images = %d' % train_sample)
        
    elif opt.dataset_mode =='gokberk_fr':
        #target_file='/auto/data2/salman/datasets/recsynth/IXI_gaussian/T1_'+str(opt.us_t1)+'_multi_synth_recon_'+str(opt.phase)+'.mat'
        #target_file = '/auto/data2/gelmas/deneme_MRNet/test/data.mat'
        # a = np.arange(opt.slice_num)
        # a = np.arange(20)
        # np.random.shuffle(a)
        # print(a)
        # return 0
        
        if str(opt.phase) != "test":
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/'+opt.dataset_name+'_usx6/'+str(opt.phase)+'/data.mat'
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/fake_superres/'+opt.us_pattern+'/IXI_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_IXI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_IXI=data_us_IXI.astype(np.float32)
            us_mask_IXI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_IXI=us_mask_IXI.astype(np.float32)  
            data_fs_IXI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_IXI=data_fs_IXI.astype(np.float32) 
            
            if str(opt.phase) == "train":
                delete_vec_tr = np.arange(880)
                np.random.shuffle(delete_vec_tr)
                fake_data_us_IXI = np.delete(data_us_IXI,  delete_vec_tr, axis = 1)
                fake_data_fs_IXI = np.delete(data_fs_IXI,  delete_vec_tr, axis = 1)
                fake_us_mask_IXI = np.delete(us_mask_IXI,  delete_vec_tr, axis = 1)
            elif str(opt.phase) == "val":
                delete_vec_val = np.arange(247)
                np.random.shuffle(delete_vec_val)
                print(delete_vec_val)
                fake_data_us_IXI = np.delete(data_us_IXI,  delete_vec_val, axis = 1)
                fake_data_fs_IXI = np.delete(data_fs_IXI,  delete_vec_val, axis = 1)
                fake_us_mask_IXI = np.delete(us_mask_IXI,  delete_vec_val, axis = 1)
            print(fake_data_us_IXI.shape)
            print(fake_data_fs_IXI.shape) 
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/IXI_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_IXI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_IXI=data_us_IXI.astype(np.float32)
            us_mask_IXI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_IXI=us_mask_IXI.astype(np.float32)  
            data_fs_IXI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_IXI=data_fs_IXI.astype(np.float32) 
            
            data_us_IXI = np.concatenate((data_us_IXI,fake_data_us_IXI),axis=1 )
            data_fs_IXI = np.concatenate((data_fs_IXI,fake_data_fs_IXI),axis=1 )
            us_mask_IXI = np.concatenate((us_mask_IXI,fake_us_mask_IXI),axis=1 )
            
            
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/fake_superres/'+opt.us_pattern+'/fastMRI_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_fastMRI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_fastMRI=data_us_fastMRI.astype(np.float32)
            us_mask_fastMRI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_fastMRI=us_mask_fastMRI.astype(np.float32)  
            data_fs_fastMRI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_fastMRI=data_fs_fastMRI.astype(np.float32)  
            
            if str(opt.phase) == "train":
                delete_vec_tr = np.arange(880)
                np.random.shuffle(delete_vec_tr)
                fake_data_us_fastMRI = np.delete(data_us_fastMRI,  delete_vec_tr, axis = 1)
                fake_data_fs_fastMRI = np.delete(data_fs_fastMRI,  delete_vec_tr, axis = 1)
                fake_us_mask_fastMRI = np.delete(us_mask_fastMRI,  delete_vec_tr, axis = 1)
            elif str(opt.phase) == "val":
                delete_vec_val = np.arange(247)
                np.random.shuffle(delete_vec_val)
                fake_data_us_fastMRI = np.delete(data_us_fastMRI,  delete_vec_val, axis = 1)
                fake_data_fs_fastMRI = np.delete(data_fs_fastMRI,  delete_vec_val, axis = 1)
                fake_us_mask_fastMRI = np.delete(us_mask_fastMRI,  delete_vec_val, axis = 1)
            print(fake_data_us_fastMRI.shape)
            print(fake_data_fs_fastMRI.shape) 
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/fastMRI_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_fastMRI=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_fastMRI=data_us_fastMRI.astype(np.float32)
            us_mask_fastMRI=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_fastMRI=us_mask_fastMRI.astype(np.float32)  
            data_fs_fastMRI=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_fastMRI=data_fs_fastMRI.astype(np.float32)  
            
            data_us_fastMRI = np.concatenate((data_us_fastMRI,fake_data_us_fastMRI),axis=1 )
            data_fs_fastMRI = np.concatenate((data_fs_fastMRI,fake_data_fs_fastMRI),axis=1 )
            us_mask_fastMRI = np.concatenate((us_mask_fastMRI,fake_us_mask_fastMRI),axis=1 )
            
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/fake_superres/'+opt.us_pattern+'/brats_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_brats=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_brats=data_us_brats.astype(np.float32)
            us_mask_brats=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_brats=us_mask_brats.astype(np.float32)  
            data_fs_brats=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_brats=data_fs_brats.astype(np.float32) 
            
            
            
            if str(opt.phase) == "train":
                delete_vec_tr = np.arange(880)
                np.random.shuffle(delete_vec_tr)
                fake_data_us_brats = np.delete(data_us_brats,  delete_vec_tr, axis = 1)
                fake_data_fs_brats = np.delete(data_fs_brats,  delete_vec_tr, axis = 1)
                fake_us_mask_brats = np.delete(us_mask_brats,  delete_vec_tr, axis = 1)
            elif str(opt.phase) == "val":
                delete_vec_val = np.arange(247)
                np.random.shuffle(delete_vec_val)
                fake_data_us_brats = np.delete(data_us_brats,  delete_vec_val, axis = 1)
                fake_data_fs_brats = np.delete(data_fs_brats,  delete_vec_val, axis = 1)
                fake_us_mask_brats = np.delete(us_mask_brats,  delete_vec_val, axis = 1)
            print(fake_data_us_brats.shape)
            print(fake_data_fs_brats.shape)  
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/420_slices/'+opt.us_pattern+'/brats_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            print(target_file)
            f = h5py.File(target_file,'r') 
            data_us_brats=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us_brats=data_us_brats.astype(np.float32)
            us_mask_brats=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask_brats=us_mask_brats.astype(np.float32)  
            data_fs_brats=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs_brats=data_fs_brats.astype(np.float32) 
            
            data_us_brats = np.concatenate((data_us_brats,fake_data_us_brats),axis=1 )
            data_fs_brats = np.concatenate((data_fs_brats,fake_data_fs_brats),axis=1 )
            us_mask_brats = np.concatenate((us_mask_brats,fake_us_mask_brats),axis=1 )
            
            data_us = np.concatenate((data_us_IXI,data_us_fastMRI,data_us_brats),axis=1 )
            us_mask = np.concatenate((us_mask_IXI,us_mask_fastMRI,us_mask_brats),axis=1 )
            data_fs = np.concatenate((data_fs_IXI,data_fs_fastMRI,data_fs_brats),axis=1 )
            
            shuff_vec = np.arange(opt.slice_num)
            np.random.shuffle(shuff_vec)
            data_us = data_us[:,shuff_vec,:,:]
            data_fs = data_fs[:,shuff_vec,:,:]
            us_mask = us_mask[:,shuff_vec,:,:]
            # print(np.array_equal(np.concatenate((data_us_IXI,data_us_fastMRI,data_us_brats),axis=1 ),data_us))
            
        else:
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/420_slices/'+opt.dataset_name+'_usx6/'+str(opt.phase)+'/data.mat'
            target_file = '/auto/data2/gelmas/brain_dataset/cleared/umram/'+opt.us_pattern+"/"+opt.dataset_name+'_usx'+opt.acc_rate+'/'+str(opt.phase)+'/data.mat'
            # target_file = '/auto/data2/gelmas/brain_dataset/cleared/fake_420_slices/'+opt.us_pattern+"/"+opt.dataset_name+'_usx'+opt.acc_rate+'/'+"val"+'/data.mat'
            print(target_file)
            #target_file = '/auto/data2/gelmas/recon_rGAN_fastMRI_knee/'+str(opt.phase)+'/data.mat' 
     	#target_file = '/auto/data2/gelmas/recon_rGAN_brats/'+str(opt.phase)+'/data.mat' 
    
    
            #target_file='/auto/data2/korkmaz/DIP_PROJECT/fastmri/brain/'+opt.output_mod+'/'+opt.output_mod+'_under_sampled_'+str(opt.us_t1)+'x_multicoil_'+str(opt.phase)+'.mat'
            f = h5py.File(target_file,'r') 
            data_us=np.transpose(np.array(f['data_us']),(0,1,3,2))
            data_us=data_us.astype(np.float32)
            us_mask=np.expand_dims(np.transpose(np.array(f['us_map']),(0,2,1)),axis=0)
            us_mask=us_mask.astype(np.float32)  
            #data_fs=np.transpose(np.array(f['images_fs']),(0,1,3,2))
            #data_fs=data_fs.astype(np.float32)
    
            data_fs=np.expand_dims(np.transpose(np.array(f['data_fs']),(0,2,1)),axis=0)
            data_fs=data_fs.astype(np.float32)  
        print(data_us.shape)
        print(us_mask.shape)
        print(data_fs.shape)  
        # ind = 0
        # print(np.amax( np.sqrt(np.square(data_us[0,ind,:,:]) + np.square(data_us[1,ind,:,:]) ) ), " ", np.amin( np.sqrt(np.square(data_us[0,ind,:,:]) + np.square(data_us[1,ind,:,:]) ) ))
        # print(np.amax( data_fs[0,ind,:,:]), " ", np.amin( data_fs[0,ind,:,:]))
        # print(np.amax( us_mask[0,ind,:,:]), " ", np.amin( us_mask[0,ind,:,:]))
        # return 0
        dataset=[]                       
        # for train_sample in range(data_us.shape[1]):
        for train_sample in range(opt.slice_num):
        # for train_sample in range(357):
            print("US BEFORE MIN: ", np.min(data_us[:,train_sample,:,:]))
            print("US BEFORE MAX: ", np.max(data_us[:,train_sample,:,:]))
            data_us[:,train_sample,:,:]=(data_us[:,train_sample,:,:]-0.5)/0.5
            print("US AFTER MIN: ", np.min(data_us[:,train_sample,:,:]))
            print("US AFTER MAX: ", np.max(data_us[:,train_sample,:,:]))
            temp = us_mask[:,train_sample,:,:].reshape(256,256)
 	    #print(us_mask.shape)
 	    #print(asd)	
            # print("us: ",np.amax( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) ), np.amin( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) ), " fs: ",np.amax( data_fs[0,train_sample,:,:]), np.amin( data_fs[0,train_sample,:,:]))
            # if np.isnan(np.amax( np.sqrt(np.square(data_us[0,train_sample,:,:]) + np.square(data_us[1,train_sample,:,:]) ) )):
            #     print(train_sample)
                
            temp = np.fft.ifftshift(temp)
            us_mask[:,train_sample,:,:]=temp
            print("FS BEFORE MIN: ", np.min(data_fs[:,train_sample,:,:]))
            print("FS BEFORE MAX: ", np.max(data_fs[:,train_sample,:,:]))
            data_fs[:,train_sample,:,:]=(data_fs[:,train_sample,:,:]-0.5)/0.5
            print("FS AFTER MIN: ", np.min(data_fs[:,train_sample,:,:]))
            print("FS AFTER MAX: ", np.max(data_fs[:,train_sample,:,:]))
            dataset.append({'A': torch.from_numpy(data_us[:,train_sample,:,:]), 'B':torch.from_numpy(data_fs[:,train_sample,:,:]),
                      'C': torch.from_numpy(us_mask[:,train_sample,:,:]),      
            'A_paths':opt.dataroot, 'B_paths':opt.dataroot, 'C_paths':opt.dataroot})
        # return 0
        print('#training images = %d' % train_sample)
          
         

    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data
