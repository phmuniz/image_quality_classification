from torchvision import models
from models.resnet import MyResnet

def set_model(model_name):

    if model_name == 'resnet':
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model = MyResnet(resnet, 2)

    return model