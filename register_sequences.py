import os

# TODO: Replace with the path to the dataset's root directory
dataset_root_path = "/path/to/dataset/root/"

# TODO: Replace with the path to the output directory for the registered volumes
output_root_path = "/path/to/output/root/"

# TODO: Customize the MRI sequence names as per your dataset
sequences = {
    'T1': 'T1w',  # Example: 'T1w' or 'T1_weighted'
    'T2': 'T2w',  # Example: 'T2w' or 'T2_weighted'
    'FLAIR': 'FLAIR',  # Example: 'FLAIR' or 'PD' or 'T2_FLAIR'
    # Add or remove sequences as needed
}

# Ensure the output root path exists
os.makedirs(output_root_path, exist_ok=True)

# Get a list of all subject folders in the dataset
subject_folders = [f for f in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, f))]
total = len(subject_folders)
current_subject_number = 1

# Iterate over each subject folder
for subject_folder in subject_folders:
    print(f"Processing {current_subject_number} of {total} subjects.")
    current_subject_number += 1
    
    # Full path to the current subject's folder
    subject_path = os.path.join(dataset_root_path, subject_folder)
    
    # Full path to the current subject's output folder
    subject_output_path = os.path.join(output_root_path, subject_folder)
    os.makedirs(subject_output_path, exist_ok=True)
    
    # List all items in the current subject's folder
    items_in_subject_folder = os.listdir(subject_path)
    
    # Check for the presence of sequence files and perform registration
    for item in items_in_subject_folder:
        # Construct the full path to the current item
        item_path = os.path.join(subject_path, item)
        
        # Check if the item is a file and matches the expected pattern
        if os.path.isfile(item_path) and sequences['T1'] in item and item.endswith('.nii.gz'):
            t1_file_path = item_path
            
            # Loop over each sequence to find and register it to the T1 sequence
            for seq_name, seq_identifier in sequences.items():
                if seq_name == 'T1':
                    continue  # Skip the T1 sequence as it is the reference
                seq_file_name = item.replace(sequences['T1'], seq_identifier)
                seq_file_path = os.path.join(subject_path, seq_file_name)
                
                if os.path.isfile(seq_file_path):
                    print(f"Registering {seq_file_name} to {item}")
                    # Output paths for registered volume and matrix
                    trans_add = os.path.join(subject_output_path, seq_file_name.replace('.nii.gz', '_reg.nii.gz'))
                    mat_add = os.path.join(subject_output_path, seq_file_name.replace('.nii.gz', '_reg.mat'))
                    
                    # TODO: Customize the registration command as needed
                    command = f"fsl5.0-flirt -in {seq_file_path} -ref {t1_file_path} -out {trans_add} -omat {mat_add} -bins 256 -cost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12 -interp spline"
                    os.system(command)
                    print("Registration complete.\n")
            
            # Add additional sequence registration logic here if needed

print("Image registration for all subjects completed.")
