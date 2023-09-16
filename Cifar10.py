
from typing import Callable, Optional
from torchvision import datasets
from PIL import Image

from data_loader import load_training, load_testing, load_validation, load_meta

class Cifar10(datasets.VisionDataset):

    '''
    
    To be used as a direct replacement of torchvision.datasets.CIFAR10
    in order to use custom data splitting
    
    '''

    def __init__(self, train: bool = True, transform: Optional[Callable] = None) -> None:
        super().__init__(None, transform=transform)

        self.data = None
        self.targets = None

        self.data, self.targets = load_training() if train else load_testing()

        # reshape for torch image
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1)) # convert to color last channel format

        self.class_index_mapping = {_class: i for i, _class in enumerate(load_meta())}


    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> tuple:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    

    def get_classes(self) -> dict:
        'returns mapping of class name to label'
        return self.class_index_mapping

        

