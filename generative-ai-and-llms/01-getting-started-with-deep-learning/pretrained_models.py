import torch
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

model.to(device)

pre_trans = weights.transforms()
pre_trans

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

show_image("data/doggy_door_images/happy_dog.jpg")

def load_and_process_image(file_path):
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(file_path).shape)
    
    image = tv_io.read_image(file_path).to(device)
    image = pre_trans(image)  # weights.transforms()
    image = image.unsqueeze(0)  # Turn into a batch
    return image

processed_image = load_and_process_image("data/doggy_door_images/happy_dog.jpg")
print("Processed image shape: ", processed_image.shape)

import torchvision.transforms.functional as F

plot_image = F.to_pil_image(torch.squeeze(processed_image))
plt.imshow(plot_image, cmap='gray')

vgg_classes = json.load(open("data/imagenet_class_index.json"))

vgg_classes["0"]

def readable_prediction(image_path):
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    output = model(image)[0]  # Unbatch
    predictions = torch.topk(output, 3)
    indices = predictions.indices.tolist()
    # Print predictions in readable form
    out_str = "Top results: "
    pred_classes = [vgg_classes[str(idx)][1] for idx in indices]
    out_str += ", ".join(pred_classes)
    print(out_str)

    return predictions

# readable_prediction("data/doggy_door_images/happy_dog.jpg")
# readable_prediction("data/doggy_door_images/brown_bear.jpg")
# readable_prediction("data/doggy_door_images/sleepy_cat.jpg")

import numpy as np

def doggy_door(image_path):
    show_image(image_path)
    image = load_and_process_image(image_path)
    idx = model(image).argmax(dim=1).item()
    print("Predicted index:", idx)
    if 151 <= idx <= 268:
        print("Doggy come on in!")
    elif 281 <= idx <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")

doggy_door("data/doggy_door_images/brown_bear.jpg")
# doggy_door("data/doggy_door_images/happy_dog.jpg")
# doggy_door("data/doggy_door_images/sleepy_cat.jpg")

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)