import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image

# Load your model (adjust path and model architecture)
model = models.resnet50(pretrained=False)  # Example architecture
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Replace num_classes with your number of bird classes
model.load_state_dict(torch.load("bird_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names (replace with your bird species list)
class_names = ["Sparrow", "Eagle", "Blue Jay"]  # Example

def classify_bird(image):
    # Convert Gradio image (PIL) to tensor
    img = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
    return class_names[predicted_idx]

# Create Gradio interface
interface = gr.Interface(
    fn=classify_bird,
    inputs=gr.Image(type="pil", label="Upload a bird image"),
    outputs=gr.Textbox(label="Predicted Bird Species"),
    title="Bird Classification App",
    description="Upload an image of a bird to classify its species."
)

# Launch the app
interface.launch()
