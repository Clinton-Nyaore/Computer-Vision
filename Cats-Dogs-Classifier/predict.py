import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T


# Lets load our saved model
def load_model():
    checkpoint = torch.load('new_alexnetcatsdogs.pth')
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.avgpool = nn.AdaptiveAvgPool2d((6, 6))
    return model


# Preprocess the image as before
def process_image(image_path):
    transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.open(image_path)
    image = transform(image)
    image = torch.unsqueeze(torch.Tensor(image), 0)

    return image


# Make predictions
def prediction(image_path, model):
    model.eval()
    image = image_path
    outputs = model(image)
    _, ps = torch.max(outputs.data, 1)
    class_labels = ['Cat', 'Dog']

    return class_labels[int(ps)]


def main():
    model = load_model()
    image_path = 'E:\MyNotesCertsProjects\Projects\DLModels\CatsDogsClassifier\images\img001.jfif'
    image = process_image(image_path=image_path)
    preds = prediction(image_path=image, model=model)
    print(preds)

    return None


