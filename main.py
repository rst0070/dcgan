from tqdm import tqdm
from Dataset import FaceData
from Trainer import Trainer
from Discriminator import Discriminator
from Generator import Generator
from torchvision.utils import save_image

class Main:
    
    def __init__(self):
        
        dataset = FaceData(dataroot="D:/dataset/celeba", img_size=64)
        D = Discriminator().cuda()
        G = Generator().cuda()
        self.trainer = Trainer(dataset=dataset, generator=G, discriminator=D)
        
    def saveImgs(self, img_list, start):
        
        idx = start
        for imgs in tqdm(img_list, desc="saveing output"):
            save_image(tensor=imgs[0], fp=f"output/{idx}.png")
            idx += 1
        return idx
        
    def start(self):
        
        img_list = []
        
        start_idx = 0
        for epoch in range(1, 10):
            
            img_list = img_list + self.trainer.train()
            start_idx = self.saveImgs(img_list, start_idx)
        
if __name__ == "__main__":
    
    main = Main()
    main.start()