from abc import ABC, abstractmethod

class Augmentation(ABC):
    @abstractmethod
    def get_transforms(self):
        pass


class TrainAugmentation(Augmentation):

    def get_transforms(self):
        pass


class ValidAugmentation(Augmentation):

    def get_transforms(self):
        pass

def get_transforms_obj(split):
    if split.lower() == 'train':
        return TrainAugmentation()
    else:
        return ValidAugmentation()