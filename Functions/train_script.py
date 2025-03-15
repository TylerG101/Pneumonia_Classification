import os
from pathlib import Path
import torch
import torchvision
from torchvision.transforms import v2
import data_setup
import trainer
import save_model


#Set hyperparameters
NUM_EPOCHS = 8
BATCH_SIZE = 32
LEARNING_RATE = 0.002


#Set up directories
image_path = Path("../Data/chest_xray")
train_dir = image_path / "train"
test_dir = image_path / "test"


#Set device
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"


###We will use the model backbone AlexNet from torchvision, download it and its pretrained weights, and send them to device
weights = torchvision.models.AlexNet_Weights.DEFAULT
model = torchvision.models.alexnet(weights=weights).to(device)
#model          ###uncomment to inspect model architecture


###We also need the transforms performed on data input into AlexNet
#Include data augmentation in the form of random rotation to help combat overfitting
augment_transform = v2.Compose([
    v2.Resize([224,224]),
    v2.RandomRotation(degrees=(0,360)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])

])


###Prepare training and testing dataloaders using this transform
train_DL, test_DL, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                               test_dir=test_dir,
                                                               transform=augment_transform,
                                                               batch_size=BATCH_SIZE)


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


###Define our loss and optimizer functions
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            weight_decay=0.01,
                            lr=LEARNING_RATE)


###Train the model
import trainer
model = model.to(device)            #Make sure the model and all components are on target device

results = trainer.train(model=model,
                       train_dataloader=train_DL,
                    test_dataloader=test_DL,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    epochs=NUM_EPOCHS,
                    device=device)


###Save the model
save_model.save_model(model=model,
                      target_dir="Models",
                      model_name="Pneumonia_Class_Model.pth")
