from torchvision import models
from models.resnet import MyResnet
from models.mobilenet import MyMobilenet

def set_model(model_name):

    if model_name == 'resnet':
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model = MyResnet(resnet, 2)

    elif model_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        model = MyMobilenet(mobilenet, 2)

    return model