from abc import ABC, abstractmethod
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Augmentation(ABC):
    @abstractmethod
    def get_transforms(self, height=512, width=512):
        pass


class TrainAugmentation(Augmentation):

    def get_transforms(self, height=512, width=512):
      return A.Compose([
            A.Resize(height=height, width=width, p=1),
            A.RandomRotate90(),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.7),                    
            # A.OneOf([
            #     A.MotionBlur(p=0.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            #             ], p=0.5),                
            # A.OneOf([                        
            #     A.IAASharpen(),
            #     A.IAAEmboss(),
            #     A.RandomBrightnessContrast(),
            #             ], p=0.5),
            A.Cutout(num_holes=8,  max_h_size=20, max_w_size=20, p=0.7),
            A.RandomShadow(p=0.2),
            ToTensorV2(p=1.0)
        ])
        


class ValidAugmentation(Augmentation):

    def get_transforms(self, height=512, width=512):
        return A.Compose([
            A.Resize(height=height, width=width, p=1),
            ToTensorV2(p=1.0)
        ])

def get_transforms_obj(split):
    if split.lower() == 'train':
        return TrainAugmentation().get_transforms()
    else:
        return ValidAugmentation().get_transforms()