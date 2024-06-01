import os
import pandas as pd

# Load the CSV data
df = pd.read_csv('results.tsv', delimiter='\t')

# Find the filenames with 'small_tissue_filled_max_area' > 100;blurry_removed_num_regions > 1
filtered_filenames = df[df['blurry_removed_num_regions'] > 1]['filename'].tolist()

# Define the dataset directory
dataset_dir = "IHC/train"
count = 0
# Iterate over the files in the dataset directory
for filename in os.listdir(dataset_dir):
    print(filename)
    # If the file is in the list of files to remove
    if filename in filtered_filenames:
        # Construct the full file path
        file_path = os.path.join(dataset_dir, filename)
        # Remove the file
        os.remove(file_path)
        count += 1
        print(f'Removed {filename}', count)
