import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torchvision

class ImgTrainTransform:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.aug = A.Compose([
            A.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)}, p=0.25),
            A.Resize(size[0], size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Normalize(mean=self.normalization[0], std=self.normalization[1]),
            ToTensorV2(),
        ])

    def __call__(self, img):   
        img = self.aug(image=np.array(img))['image']        
        return img
        


class ImgEvalTransform:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)