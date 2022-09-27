import os
import requests
import torch
from torchvision import transforms
import numpy as np
import unet


class CatDogUNet:
    def __init__(self):
        filename = "unet_model.ckpt"
        if not os.path.exists(filename):
            model_path = os.path.join(
                "https://connectionsworkshop.blob.core.windows.net/pets", filename
            )
            r = requests.get(model_path)
            with open(filename, "wb") as outfile:
                outfile.write(r.content)
        self.model = unet.CatDogUNet(num_classes=1)
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()

    def predict(self, image):
        # transform input image (as required by model)
        transform_input = transforms.Compose([transforms.Resize((192, 192)),])
        image = image.values
        image = image[:, :, 0:3]  # make sure we have only 3 channels
        image = np.transpose(image, (2, 0, 1))
        image = image / 255
        image = torch.from_numpy(image).type(torch.float32)
        image = transform_input(image)

        # make prediction
        prediction = self.model(image)
        return prediction
