from PIL import Image
import timm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_image(image_path):
    """Load image from the local file system and convert to RGB if required."""
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        image = image.convert("RGB")
    return image


def apply_gradcam(model, input_tensor, target_layer):
    # Register hooks to capture gradients and activations
    activations = []
    gradients = []

    # Define forward hook
    def forward_hook(module, input, output):
        activations.append(output)

    # Define backward hook
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Attach hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Perform a forward pass
    output = model(input_tensor)

    # Compute gradients with respect to the desired class (index 0 for binary classification)
    model.zero_grad()
    class_idx = output.argmax().item()
    output[:, class_idx].backward()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Extract gradients and activations
    gradients = gradients[0]  # Shape: [batch_size, num_channels, height, width]
    activations = activations[0]  # Shape: [batch_size, num_channels, height, width]

    if len(gradients.shape) < 4 or len(activations.shape) < 4:
        raise ValueError("Grad-CAM requires gradients of shape [batch_size, num_channels, height, width]")

    # Compute the Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    gradcam = torch.sum(weights * activations, dim=1).squeeze()

    # Apply ReLU and normalize
    gradcam = F.relu(gradcam)
    gradcam = gradcam / gradcam.max()

    # Convert to numpy for visualization
    gradcam_np = gradcam.cpu().detach().numpy()
    return gradcam_np


def visualize_gradcam(image_path, gradcam, alpha=0.5, threshold=0.3):
    image = load_image(image_path)
    image = np.array(image)

    # Resize Grad-CAM to match the original image size
    gradcam_resized = cv2.resize(gradcam, (image.shape[1], image.shape[0]))

    # Threshold the heatmap
    gradcam_resized[gradcam_resized < threshold] = 0  # Apply thresholding

    # Overlay the Grad-CAM heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # Show the image with Grad-CAM overlay
    plt.imshow(overlay)
    plt.axis('off')
    plt.show()



def predict_cancer(image_path, threshold=0.5, visualize=False):
    # Load model
    model = timm.create_model(
        model_name="hf-hub:1aurent/resnet50.tcga_brca_simclr",
        pretrained=True,
    ).eval()

    # Load the local image
    image = load_image(image_path)

    # Get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Preprocess and pass the image through the model
    input_tensor = transforms(image).unsqueeze(0)
    with torch.no_grad():
        features = model(input_tensor)

    # Choose a convolutional layer for Grad-CAM visualization
    target_layer = model.layer4[2].conv3  # Modify the target layer based on model architecture

    if visualize:
        gradcam = apply_gradcam(model, input_tensor, target_layer)
        visualize_gradcam(image_path, gradcam)

    # Apply Global Average Pooling (1D) and sigmoid to get a single value prediction
    pooled_features = torch.mean(features, dim=1)
    prediction = torch.sigmoid(pooled_features)

    # Check if the output is greater than or equal to threshold
    if prediction.item() >= threshold:
        return "Cancer detected"
    else:
        return "No cancer detected"


# Replace with the path to your image file
image_path = r"C:\Users\Iqqu\Desktop\pythonProject1\pythonProject1\carebreast\static\images\42rmzk0j.png"
result = predict_cancer(image_path, visualize=True)
print(result)
