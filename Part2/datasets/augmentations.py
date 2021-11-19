from abc import ABC, abstractmethod
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Augmentation(ABC):
    @abstractmethod
    def get_transforms(self, height=512, width=512):
        return A.Compose([
            A.Resize(height=height, width=width, p=1),
            A.RandomRotate90(),
            A.Flip(p=0.5),
            A.Transpose(),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.7),                    
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                        ], p=0.7), #0.7                    
            A.OneOf([                        
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
                        ], p=0.7), #0.7
            A.Cutout(num_holes=8,  max_h_size=20, max_w_size=20, p=0.7),
            A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=3, p=0.2),
            A.RandomFog(fog_coef_lower=0., fog_coef_upper=0.5, alpha_coef=0.1, p=0.3),
            ToTensorV2(p=1.0)
        ])


class TrainAugmentation(Augmentation):

    def get_transforms(self, height=512, width=512):
        return A.Compose([
            A.Resize(height=height, width=width, p=1),
            ToTensorV2(p=1.0)
        ])


class ValidAugmentation(Augmentation):

    def get_transforms(self):
        return None

def get_transforms_obj(split):
    if split.lower() == 'train':
        return TrainAugmentation().get_transforms()
    else:
        return ValidAugmentation().get_transforms()