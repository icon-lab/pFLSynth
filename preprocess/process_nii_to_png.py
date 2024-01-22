import os
import nibabel as nib
import matplotlib.pyplot as plt

def process_nii_to_png(input_dir, output_dir):
    """
    Converts NIfTI files to PNG format, saving each slice of the 3D image as a separate 2D image.

    :param input_dir: Directory containing NIfTI files.
    :param output_dir: Directory where PNG files will be saved.
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".nii") or file.endswith(".nii.gz"):
                file_path = os.path.join(root, file)
                img = nib.load(file_path)
                data = img.get_fdata()

                # Create a subdirectory for each NIfTI file's slices
                slice_dir = os.path.join(output_dir, os.path.splitext(file)[0])
                os.makedirs(slice_dir, exist_ok=True)

                for slice_idx in range(data.shape[2]):
                    slice_file = os.path.join(slice_dir, f"slice_{slice_idx}.png")
                    plt.imsave(slice_file, data[:, :, slice_idx], cmap='gray')

if __name__ == "__main__":
    # TODO: Set your input directory containing NIfTI files
    # Hint: This should be the same as the output directory from register_sequences.py
    input_directory = "/path/to/input/directory"

    # TODO: Set your output directory for PNG files
    output_directory = "/path/to/output/directory"

    process_nii_to_png(input_directory, output_directory)
