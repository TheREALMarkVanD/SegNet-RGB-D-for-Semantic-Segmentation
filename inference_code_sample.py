import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from segnet2 import SegNet  # Import SegNet class from segnet.py

class SegNetInference:
    def __init__(self, model_weights_path, in_channels, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SegNet(in_channels, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            Resize((480, 640)),
            transforms.ToTensor(),
            
        ])

    def preprocess_image(self, image_path, depth_path):
        image = Image.open(image_path).convert('RGB')
        depth_image = Image.open(depth_image_path)
        # Convert them to PyTorch tensors
        rgb_tensor = self.transform(image)
        depth_tensor = self.transform(depth_image)
        # Stack them along the channel dimension
        stacked_tensor = torch.cat((rgb_tensor, depth_tensor), dim=0)
        return stacked_tensor.unsqueeze(0).to(self.device)

    def infer(self, image_path, depth_path):
        image = self.preprocess_image(image_path, depth_path)
        with torch.no_grad():
            output = self.model(image)
            # Apply softmax to convert logits to probabilities
            probabilities = torch.softmax(output, dim=1).cpu() 
            predicted_class = torch.argmax(probabilities, dim=1).squeeze().cpu().numpy()
        return predicted_class

# Load the trained model weights
model_weights_path = "path_to_your_trained_model_weights.pth"
in_channels = 4  # Assuming you have 4 input channels (RGBD)
num_classes = 895  # Adjust based on your dataset
segnet_inference = SegNetInference(model_weights_path, in_channels, num_classes)

# Perform inference on a single image
image_path = "path_to_your_image.jpg"
depth_path = "path_to_your_depth.jpg"
predicted_class = segnet_inference.infer(image_path, depth_path)

# Visualize the predicted segmentation mask and probabilities
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(predicted_class, cmap='jet', vmin=0, vmax=num_classes-1)
plt.colorbar()
plt.title("Segmentation Mask")

plt.subplot(1, 2, 2)
plt.imshow(probabilities[0, 0, :, :], cmap='jet', vmin=0, vmax=1)
plt.colorbar()
plt.title("Probability Map for Class 0")  # Adjust the class index as needed
plt.show()

# adjust the 2 paths, and in channels and num classes as required

