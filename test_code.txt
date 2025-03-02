import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from SegNet_model_code import SegNet  # Import SegNet class from segnet.py

class NYUDataset(Dataset):
    def __init__(self, image_with_depth_path, label_path, transform=None):
        self.images_with_depth =  torch.load( image_with_depth_path)
        self.labels =  torch.load(label_path)
        self.transform = transform  # Optional transformation for data augmentation

    def __len__(self):
        return len(self.images_with_depth)

    def __getitem__(self, idx):
        image_with_depth = self.images_with_depth[idx]
        label = self.labels[idx]
        if self.transform:
            image_with_depth, label = self.transform(image_with_depth, label)
        return image_with_depth, label

# Define a function to calculate Intersection over Union (IoU)
def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Define a function to calculate Dice coefficient
def calculate_dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice_coefficient = (2. * intersection) / union
    return dice_coefficient

# Define a function to calculate pixel accuracy
def calculate_pixel_accuracy(y_true, y_pred):
    correct_pixels = np.sum(y_true == y_pred)
    total_pixels = y_true.size
    pixel_accuracy = correct_pixels / total_pixels
    return pixel_accuracy

'''
def calculate_class_accuracy(y_true, y_pred, num_classes):
    class_accuracies = []
    for c in range(num_classes):
        true_class = (y_true == c)
        pred_class = (y_pred == c)
        tp = np.sum(true_class & pred_class)
        fn = np.sum(true_class & ~pred_class)
        if np.sum(true_class) == 0:
            class_accuracies.append(np.nan)
        else:
            class_accuracy = tp / (tp + fn)
            class_accuracies.append(class_accuracy)
    return class_accuracies

'''

# Iterate over the test dataset and make predictions
iou_scores = []
dice_coefficients = []
pixel_accuracies = []

# Load the test dataset
test_image_with_depth_path = Path('test_image_with_depth_tensor.pt')
test_label_path = Path('test_label_tensor.pt')

test_dataset = NYUDataset(test_image_with_depth_path, test_label_path)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
#model = SegNet()  # Initialize your model
model.load_state_dict(torch.load('checkpoints/best_model.pth'))  # Load the best model
model.to(device)
model.eval()  # Set the model to evaluation mode
test_dataset.labels = test_dataset.labels.to(torch.long)

# Define a function to visualize the predictions
def visualize_predictions(image, label, prediction):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='jet')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='jet')
    plt.title("Predicted Segmentation Mask")
    plt.axis('off')
    
    plt.show()

# Iterate over the test dataset and make predictions
for i, (image_with_depth, label) in enumerate(test_loader):
    image_with_depth = image_with_depth.to(device)
    #label = label.to(device) 

    # Forward pass
    with torch.no_grad():
        output = model(image_with_depth)
        output_softmax = F.softmax(output, dim=1) #probabilities
        predictions = torch.argmax(output_softmax, dim=1)
    # Convert tensor to numpy array
    image_with_depth_np = image_with_depth.squeeze().permute(1, 2, 0).cpu().numpy()
    label_np = label.squeeze().cpu().numpy()
    prediction_np = predictions.squeeze().cpu().numpy()

    iou_score = calculate_iou(label_np, prediction_np)
    dice_coefficient = calculate_dice_coefficient(label_np, prediction_np)
    pixel_accuracy = calculate_pixel_accuracy(label_np, prediction_np)
    #class_accuracy = calculate_class_accuracy(label_np, prediction_np, num_classes)  # Calculate class accuracy

    # Append metrics to lists
    iou_scores.append(iou_score)
    dice_coefficients.append(dice_coefficient)
    pixel_accuracies.append(pixel_accuracy)
    '''
    # Append class accuracy to corresponding class list
        for cls in range(num_classes):
            class_accuracies[cls].append(class_accuracy[cls])
    '''

    # Visualize the input image, ground truth, and predicted segmentation mask
    visualize_predictions(image_with_depth_np, label_np, prediction_np)
    input("hit enter to continue")

# Calculate average metrics
avg_iou_score = np.mean(iou_scores)
avg_dice_coefficient = np.mean(dice_coefficients)
avg_pixel_accuracy = np.mean(pixel_accuracies)
#avg_class_accuracies = [np.nanmean(acc) for acc in class_accuracies]  # Average accuracy for each class

print(f"Average IoU: {avg_iou_score:.4f}")
print(f"Average Dice Coefficient: {avg_dice_coefficient:.4f}")
print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
'''
print("Average Class Accuracies:")
for cls, acc in enumerate(avg_class_accuracies):
    print(f"Class {cls}: {acc:.4f}")
'''