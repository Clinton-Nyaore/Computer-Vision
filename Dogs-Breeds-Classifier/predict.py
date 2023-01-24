import torch
from torch import nn
from PIL import Image
import torchvision.transforms as T
from neuralnet import NeuralNetwork
# Lets load our saved model
def load_model():
    checkpoint = torch.load('newdensenetdogs.pth')
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
    class_labels = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
        'american_staffordshire_terrier', 'appenzeller',
        'australian_terrier', 'basenji', 'basset', 'beagle',
        'bedlington_terrier', 'bernese_mountain_dog',
        'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
        'bluetick', 'border_collie', 'border_terrier', 'borzoi',
        'boston_bull', 'bouvier_des_flandres', 'boxer',
        'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
        'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
        'chow', 'clumber', 'cocker_spaniel', 'collie',
        'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
        'doberman', 'english_foxhound', 'english_setter',
        'english_springer', 'entlebucher', 'eskimo_dog',
        'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
        'german_short-haired_pointer', 'giant_schnauzer',
        'golden_retriever', 'gordon_setter', 'great_dane',
        'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
        'ibizan_hound', 'irish_setter', 'irish_terrier',
        'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
        'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
        'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
        'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
        'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
        'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
        'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
        'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
        'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
        'saint_bernard', 'saluki', 'samoyed', 'schipperke',
        'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
        'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
        'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
        'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
        'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
        'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
        'west_highland_white_terrier', 'whippet',
        'wire-haired_fox_terrier', 'yorkshire_terrier']

    return class_labels[int(ps)]


def main():
    model = load_model()
    image_path = 'E:\MyNotesCertsProjects\Projects\DLModels\CatsDogsClassifier\images\img001.jfif'
    image = process_image(image_path=image_path)
    preds = prediction(image_path=image, model=model)
    print(preds)

    return None
