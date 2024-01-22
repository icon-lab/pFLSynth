import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def load_image(file_path):
    return io.imread(file_path)

def combine_contrasts(t1, t2, flair):
    combined = np.stack([(t1[..., :3].mean(axis=2)),
                         (t2[..., :3].mean(axis=2)),
                         (flair[..., :3].mean(axis=2)),
                         np.ones(t1.shape[:2])], axis=2)
    return combined

def main(input_path, output_path):
    contrasts = ['T1', 'T2', 'FLAIR']
    os.makedirs(output_path, exist_ok=True)
    files = os.listdir(os.path.join(input_path, contrasts[0]))
    s = 1
    
    for file in files:
        images = []
        # Check if all contrast images for the current slice exist
        if all(os.path.isfile(os.path.join(input_path, contrast, file)) for contrast in contrasts):
            for contrast in contrasts:
                images.append(load_image(os.path.join(input_path, contrast, file)))
            
            # Combine images and save
            combined_image = combine_contrasts(*images)
            combined_file_path = os.path.join(output_path, f"{s}.png")
            io.imsave(combined_file_path, combined_image)
            print(f"Saved combined image {s}")
            s += 1

if __name__ == '__main__':
    # TODO: Set the path where the PNG images are stored
    #Hint: This should be the same as the output directory from process_nii_to_png.py
    input_directory = "/path/to/input/directory"
    
    # TODO: Set the path where the combined images will be saved
    output_directory = "path/to/output/directory"
    
    main(input_directory, output_directory)
