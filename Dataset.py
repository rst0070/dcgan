from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

class FaceData(Dataset):
    
    def __init__(self, dataroot, img_size = 64):
        super().__init__()
        
        self.dataset = ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        return self.dataset.__getitem__(idx)