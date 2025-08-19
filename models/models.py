from torchvision import models
from models.resnet import MyResnet
from models.mobilenet import MyMobilenet
from models.vggnet import MyVGGNet

def set_model(model_name):

    if model_name == 'resnet':
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model = MyResnet(resnet, 2)
    elif model_name == 'mobilenet':
        mobilenet = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')
        model = MyMobilenet(mobilenet, 2)
    elif model_name == 'vggnet':
        vggnet = models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT')
        model = MyVGGNet(vggnet, 2)
    else:
        print("Modelo não definido. Veja as opções disponíveis ou modifique para adiconar um novo modelo.")
        exit(1)

    return model