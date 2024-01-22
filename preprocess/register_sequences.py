import os

# TODO: Replace with the path to the dataset's root directory
dataset_root_path = "/auto/data2/umirza/datasets/OASIS/OASIS_original/"

# TODO: Replace with the path to the output directory for the registered volumes
output_root_path = "/auto/data2/odalmaz/FedSynth/datasets/deneme/"

# TODO: Customize the MRI sequence names as per your dataset
sequences = {
    'T1': 'T1w',
    'T2': 'T2w',
    'FLAIR': 'FLAIR',
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
    
    # Initialize dictionary to hold the paths to each sequence file
    sequence_paths = {seq_name: '' for seq_name in sequences}
    
    # Search for sequence files in all subdirectories
    for root, dirs, files in os.walk(subject_path):
        for file in files:
            for seq_name, seq_identifier in sequences.items():
                if seq_identifier in file and file.endswith('.nii.gz'):
                    sequence_paths[seq_name] = os.path.join(root, file)
    
    # Perform registration for all sequences found
    t1_file_path = sequence_paths['T1']
    if t1_file_path:  # Proceed only if T1 sequence is found
        print(f"Found T1 sequence: {t1_file_path}")
        for seq_name, seq_file_path in sequence_paths.items():
            if seq_name == 'T1' or not seq_file_path:
                continue  # Skip if it's the T1 sequence or if the sequence file wasn't found
            print(f"Registering {seq_file_path} to T1 sequence")
            # Output paths for registered volume and matrix
            trans_add = os.path.join(subject_output_path, os.path.basename(seq_file_path).replace('.nii.gz', '_reg.nii.gz'))
            mat_add = os.path.join(subject_output_path, os.path.basename(seq_file_path).replace('.nii.gz', '_reg.mat'))
            
            # TODO: Customize the registration command as needed
            command = f"fsl5.0-flirt -in {seq_file_path} -ref {t1_file_path} -out {trans_add} -omat {mat_add} -bins 256 -cost normmi -searchrx -180 180 -searchry -180 180 -searchrz -180 180 -dof 12 -interp spline"
            os.system(command)
            print("Registration complete.\n")
    else:
        print(f"No T1 sequence found for subject {subject_folder}, skipping registration.")

print("Image registration for all subjects completed.")
