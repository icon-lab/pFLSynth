import argparse
import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

def calculate_psnr_ssim_for_image_pair(real_image_path, fake_image_path, normalize=True):
    """Calculate PSNR and SSIM for a pair of real and fake images"""
    real_image = Image.open(real_image_path).convert("L")
    fake_image = Image.open(fake_image_path).convert("L")
    
    real_image = np.asarray(real_image, dtype='float64')
    fake_image = np.asarray(fake_image, dtype='float64')

    if normalize:
        max_intensity = max(real_image.max(), fake_image.max())
        if max_intensity > 0:
            real_image /= max_intensity
            fake_image /= max_intensity

    psnr = peak_signal_noise_ratio(real_image, fake_image, data_range=1)
    ssim = structural_similarity(real_image, fake_image, data_range=1)

    return psnr, ssim

def calculate_metrics_across_subjects(fake_dir, num_im, num_im2, normalize=True):
    """Calculate metrics across subjects"""
    psnr_vals = np.zeros(len(num_im))
    ssim_vals = np.zeros(len(num_im))

    for subject_idx, (start, end) in enumerate(zip(num_im2, num_im)):
        subject_psnr_vals = []
        subject_ssim_vals = []

        for slice_idx in range(start, end + 1):
            real_image_path = os.path.join(fake_dir, f"{slice_idx}_real_B.png")
            fake_image_path = os.path.join(fake_dir, f"{slice_idx}_fake_B.png")

            if os.path.exists(real_image_path) and os.path.exists(fake_image_path):
                psnr, ssim = calculate_psnr_ssim_for_image_pair(real_image_path, fake_image_path, normalize)
                subject_psnr_vals.append(psnr)
                subject_ssim_vals.append(ssim)

        psnr_vals[subject_idx] = np.mean(subject_psnr_vals)
        ssim_vals[subject_idx] = np.mean(subject_ssim_vals)

    mean_psnr = np.mean(psnr_vals)
    std_psnr = np.std(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    std_ssim = np.std(ssim_vals)

    return mean_psnr, std_psnr, mean_ssim, std_ssim
    
# python3 evaluation.py --fake_dir /auto/data2/odalmaz/FedSynth/2_common_bidirectional/results/T1_T2_pFLSynth_common_bidirectional_true/test_latest_IXI_t1_t2/images/ --gpu_ids 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_dir', type=str, required=True, help='Path to the fake image folder')
    parser.add_argument('--normalize', type=int, default=1, help='Normalize images for PSNR/SSIM calculation')
    opt = parser.parse_args()

    # Define subject-based metrics
    num_im = [100, 205, 305, 420, 540, 641, 752, 853, 944, 1050, 1151, 1272, 1373, 1484, 1605, 1716, 1827, 1943, 2054, 2164]
    num_im2 = [1, 101, 206, 306, 421, 541, 642, 753, 854, 945, 1051, 1152, 1273, 1374, 1485, 1606, 1717, 1828, 1944, 2055]
    num_im = [x - 1 for x in num_im]
    num_im2 = [x - 1 for x in num_im2]

    # Calculate PSNR and SSIM for each subject
    mean_psnr, std_psnr, mean_ssim, std_ssim = calculate_metrics_across_subjects(opt.fake_dir, num_im, num_im2, opt.normalize)

    # Output results
    print(f"PSNR: {mean_psnr:.4g}±{std_psnr:.3g}")
    print(f"SSIM: {mean_ssim*100:.4g}±{std_ssim*100:.3g}")

    # Save metrics to a file
    metrics_file = os.path.join(os.path.dirname(opt.fake_dir), "metrics.txt")
    with open(metrics_file, "a") as myfile:
        myfile.write(f"{mean_psnr:.4g}±{std_psnr:.3g}\t{mean_ssim*100:.4g}±{std_ssim*100:.3g}\n")

    print('Metrics calculation completed.')
