import torch
import torchvision
import argparse

import data_setup

###Instantiate a parser
parser = argparse.ArgumentParser()

###Image path
parser.add_argument("--image",
                    #default="../Data/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg",
                    help="filepath to image to make predictions on")


args = parser.parse_args(args=[])


###Set up our class names
class_names = ["NORMAL", "PNEUMONIA"]


###Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


###Get image path
Img_path = "../Data/chest_xray/val/NORMAL/NORMAL2-IM-1430-0001.jpeg"

###Set model path
model_path = "Models/Pneumonia_Class_Model.pth"

###Make function to load the model (AlexNet backbone)
def load_model(model_path=model_path):
    weights = torchvision.models.AlexNet_Weights.DEFAULT
    model = torchvision.models.alexnet(weights=weights).to(device)

    ###The classifier block of AlexNet must be modified to accomodate our output shape of 2, while keeping the other block unchanged
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(in_features=9216, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(p=0.5, inplace=False),
        torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features=4096,
                        out_features=len(class_names),
                        bias=True)).to(device)



    model.load_state_dict(torch.load(model_path))

    return model


def predict_on_image(image_path=Img_path,
                     model_path=model_path):
    
    ###Load the model
    model = load_model(model_path)

    ###Read in the image
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    ###Transform the image
    image = data_setup.transform_image(image_path)
    
    image = image / 255

    ###Make a prediction
    model.eval()
    with torch.inference_mode():
        image = image.to(device)

        logits = model(image.unsqueeze(dim=0))
        label = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        label_class = class_names[label]

    print(f'Model prediction: {label_class}')

if __name__ == "__main__":
    predict_on_image()
