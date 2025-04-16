import torch
import matplotlib.pyplot as plt

def generate_images(model, epoch, test_input_noise, test_input_labels, device="cpu"):
    # Ensure model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Generate images using the model
        predictions = model(test_input_noise.to(device), test_input_labels.to(device))
        predictions = torch.round(predictions)  # Apply rounding like in TensorFlow

    # Move predictions to CPU and detach
    predictions = predictions.detach().cpu()

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, 0].numpy())  # Assuming single-channel images
        plt.axis('off')

    plt.show()
