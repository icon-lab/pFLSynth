# pFLSynth
Official Pytorch Implementation of personalized Federated Learning of MRI Synthesis(pFLSynth) which is described in the [following](https://arxiv.org/abs/2207.06509) paper:

One Model to Unite Them All: Personalized Federated Learning of Multi-Contrast MRI Synthesis. Onat Dalmaz, Usama Mirza, Gökberk Elmas, Muzaffer Özbey, Salman UH Dar, Emir Ceyani, Salman Avestimehr, Tolga Çukur. ArXiV. 2022.

<img src="main_fig.png" width="600px"/>
<img src="main_generator.png" width="600px"/>

## Dependencies
```
python>=3.6.9
torch>=1.7.1
torchvision>=0.8.2
visdom
dominate
cuda=>11.2
```
## Installation
- Clone this repo:
```bash
git clone https://github.com/icon-lab/pFLSynth
cd pFLSynth
```

# Data Preprocessing

This section details the steps required to preprocess the imaging data before it can be used for analysis. Each preprocessing step is encapsulated in its script, which can be found in the [scripts](/path/to/scripts/folder) directory.

## Registration

Image registration is a crucial step in the preprocessing pipeline, where different MRI sequences are aligned to a common space. The `register_sequences.py` script performs this task.

### Prerequisites

- FSL (FMRIB Software Library) must be installed on your system. You can download it from [FSL's official website](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki).
- Python 3.x must be installed on your system.

### Usage

To use the registration script, you need to set up the input and output paths according to your dataset's structure. Open the `register_sequences.py` script in your favorite text editor and follow the `# TODO` comments to customize the paths and sequence names.

### Demo

Here is a demo of how to configure and run the `register_sequences.py` script.

1. Set the dataset root path where your MRI sequences are located:

    ```python
    # TODO: Replace with the path to the dataset's root directory
    dataset_root_path = "/path/to/dataset/root/"
    ```

2. Set the output path where the registered volumes will be saved:

    ```python
    # TODO: Replace with the path to the output directory for the registered volumes
    output_root_path = "/path/to/output/root/"
    ```

3. Customize the sequence names based on your dataset:

    ```python
    # TODO: Customize the MRI sequence names as per your dataset
    sequences = {
        'T1': 'T1w',  # Example: 'T1w' or 'T1_weighted'
        'T2': 'T2w',  # Example: 'T2w' or 'T2_weighted'
        'T2_FLAIR': 'FLAIR',  # Example: 'FLAIR' or 'PD' or 'T2_FLAIR'
        # Add or remove sequences as needed
    }
    ```

4. Run the script:

    ```bash
    python register_sequences.py
    ```

The script will process each subject sequentially, registering the specified sequences to the T1-weighted images and saving the results in the output directory.


<!-- ## Dataset
You should structure your aligned dataset in the following way:
```
/Datasets/BRATS/
  ├── T1_T2_FLAIR
```
```
/Datasets/BRATS/T1_T2_FLAIR/
  ├── train
  ├── val  
  ├── test   
```
```
/Datasets/BRATS/T1_T2_FLAIR/train/
  ├── 0.png
  ├── 1.png
  ├── 2.png
  ├── ...
```
For instance, "0.png" looks like this:

<img src="0.png" width="600px"/>

where in the left half T1- and T2-weighted images are in the Red and Green channels respectively, and in the right half FLAIR images are in the Green channel. -->

## Federated training of pFLSynth

<br />

```
python3 /auto/data2/odalmaz/FedSynth/3_heteregeneous/github/fedsynth/train_pflsynth.py --gpu_ids 0 --dataroot Datasets/IXI/T1_T2__PD/ --dataroot2 Datasets/BRATS/T1_T2__FLAIR/ --dataroot3 Datasets/MIDAS/T1_T2/ --dataroot4 Datasets/OASIS/T1_T2__FLAIR/ --name pFLSynth_experimental_setup_3 --model pflsynth_model --which_model_netG personalized_generator --which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch --n_clients 4 --pool_size 0 --output_nc 1 --input_nc 2 --niter 75 --niter_decay 75 --save_epoch_freq 5 --checkpoints_dir checkpoints/ --federated_learning 1
```

<br />
<br />

## Inference with personalized generator


```
python3 /auto/data2/odalmaz/FedSynth/3_heteregeneous/fedsynth/test.py --dataroot /auto/data2/odalmaz/Datasets/IXI/T1_T2__PD/ --name pFLSynth_experimental_setup_3 --gpu_ids 0 --dataset_mode aligned --model pflsynth_model --which_model_netG personalized_generator --which_direction AtoB --norm batch --output_nc 1 --input_nc 2 --checkpoints_dir checkpoints/ --phase test --how_many 10000 --serial_batches --results_dir results/ --dataset_name ixi --save_folder_name IXI --n_clients 4 --task_name t1_t2
```
You can specify the site and task during inference by modifying the dataroot to match the site and changing the task name. For example, if you would like to perform inference for FLAIR->T2 task in BRATS, specify --dataroot Datasets/BRATS/T1_T2__FLAIR/ and --task_name flair_t2

# Citation
Preliminary versions of pFLSynth are presented in [MICCAI DeCaF](https://link.springer.com/chapter/10.1007/978-3-031-18523-6_8),  [NeurIPS Medical Imaging Meets](https://www.cse.cuhk.edu.hk/~qdou/public/medneurips2022/103.pdf) (Oral), and IEEE ISBI 2023.
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@misc{dalmaz2022pflsynth
  doi = {10.48550/ARXIV.2207.06509},
  
  url = {https://arxiv.org/abs/2207.06509},
  
  author = {Dalmaz, Onat and Mirza, Usama and Elmas, Gökberk and Özbey, Muzaffer and Dar, Salman UH and Ceyani, Emir and Avestimehr, Salman and Çukur, Tolga},
  
  keywords = {Image and Video Processing (eess.IV), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {One Model to Unite Them All: Personalized Federated Learning of Multi-Contrast MRI Synthesis},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
For any questions, comments and contributions, please contact Onat Dalmaz (onat[at]ee.bilkent.edu.tr) <br />

(c) ICON Lab 2022

## Acknowledgments
This code uses libraries from [pGAN](https://github.com/icon-lab/pGAN-cGAN) repository.
