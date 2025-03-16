import torch
import h5py
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
# loading data

filepath = '/content/drive/MyDrive/nyu_depth_v2_labeled.mat' #rename to your own filepath pointing at the downloaded dataset.
image_path = Path('nyu_images.npy') #for the first run of this code, these paths will be created automatically if it doesnt exist.
label_path = Path('nyu_labels.npy')
depth_path = Path('nyu_depths.npy')
image_with_depth_path=Path('image_with_depth.npy')

# Check if all .npy files exist, this code is run only once generally and can be done outside this script just to creat the .npy 


if not (image_path.exists() and label_path.exists() and depth_path.exists() and image_with_depth_path.exists()):
    # The .npy files don't exist in the first run(DO NOT CREATE PATH), so proceed with loading and processing the data

    with h5py.File(filepath, 'r') as f:

        # Access data using dataset names (replace with your desired data)
        depth_array= f['depths'][:]
        label_array= f['labels'][:]
        image_array=f['images'][:].transpose(0, 2, 3, 1) #reordered so that the coordinates of the image are (red,green,blue,image index)
        
        #Image normalization standardizes pixel values to a common range (e.g., [0,1] or [-1,1]), improving model stability, convergence, and robustness to variations in lighting and depth.
     
        normalized_images = np.zeros_like(image_array, dtype=np.float32)
        for i in range(image_array.shape[0]):
        	max_val = np.max(image_array[i])
        	if max_val != 0:
            		normalized_images[i] = image_array[i] / max_val
  
        normalized_depth = np.zeros_like(depth_array, dtype=np.float32)
        for i in range(depth_array.shape[0]):
        	max_val = np.max(depth_array[i])
        	if max_val != 0:
           		 normalized_depth[i] = depth_array[i] / max_val

        np.save(image_path, image_array)
        np.save(label_path, label_array)
        np.save(depth_path, depth_array)
        
        #Now we concatenate the Images with depth images, since loading all images at once into memory will not be possible, we concatenate them in chunks 
        chunk_size = 50

        # Get the total number of samples
        total_samples = 1449

        # Open the files
        
        image_with_depth_memmap=np.memmap(image_with_depth_path, dtype=np.float32, mode='w+', shape=(total_samples, 640, 480, 4))
        
        # Iterate over chunks of data
        for i in range(0, total_samples, chunk_size):
            # Get the current chunk indices
            start_idx = i
            end_idx = min(i + chunk_size, total_samples)

            # Concatenate image and depth channels and Write the concatenated data to the memmap array
            image_with_depth_memmap[start_idx:end_idx] = np.concatenate((image_array[start_idx:end_idx], depth_array[start_idx:end_idx][..., np.newaxis]), axis=-1)
        
        seed = 42

        # Split the data into training, validation, and testing sets with fixed split ratios and controlled randomness
        train_images, rem_images, train_labels, rem = train_test_split(image_with_depth_array, label_array, train_size=1159, shuffle=True, random_state=seed)
                val_images, test_images, val_labels, test_labels = train_test_split(rem_images, rem, test_size=145, shuffle=True, random_state=seed)
        
        np.save(image_with_depth_path, train_images)
        np.save(label_path, train_labels)

        np.save('val_image_with_depth.npy', val_images)
        np.save('val_labels.npy', val_labels)

        np.save('test_image_with_depth.npy', test_images)
        np.save('test_labels.npy', test_labels)