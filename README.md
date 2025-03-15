# Pneumonia Chest X-Ray Classification Algorithm


## Description
This model has been trained to differentiate between chest x-ray images from patients with and without pneumonia and has consistently scored an ~88% accuracy in training

## Data
Chest x-ray images comes from https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia 



## Installation and usage 
To test this model out yourself, start by simply cloning the repository and doing a 

`pip install -r requirements.txt`

From there, you can train and save a model by running 

`python train_script.py`

You can also use this model you've just trained to make a prediciton on one of the images in the predict folder by running

 `python prediciton.py` 

 (this defaults to a non-pneumonia image, but can be changed by providing a flag for a different image, like so: 

 `python prediciton.py --image ../Data/chest_xray/predict/PNEUMONIA/person1946_bacteria_4874.jpeg`  )






