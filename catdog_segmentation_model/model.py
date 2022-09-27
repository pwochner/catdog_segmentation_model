import os
import requests
import torch
from torchvision import transforms
import numpy as np
import unet


class CatDogUNet:
    def __init__(self):
        filename = "unet_model.ckpt"
        ### TODO ###
        ### Download the file containing the weights of the pre-trained model from url
        ### if the file doesn't exist already locally, and write it to a file.
        ### After that, create an instance of the model class.
        ### The code to load the model weights from file and evaluate the model is
        ### already provided.
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(self, image):
        # pre-process input image (as required by model)
        transform_input = transforms.Compose([transforms.Resize((192, 192)),])
        image = image.values
        image = image[:, :, 0:3]  # make sure we have only 3 channels
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = torch.from_numpy(image).type(torch.float32)
        image = transform_input(image)

        # make prediction
        ### TODO ###
        ### Apply the model to the pre-processed input image.
        ### Return the segmentation mask.
        return "FIX ME"

