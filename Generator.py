from turtle import forward
import torch.nn as nn

class Generator(nn.Module):
    """
    임의의 벡터 z를 입력받아 그것을 처리하여 이미지 형태의 데이터를 출력한다.  
    
    학습은 discriminator가 데이터를 학습데이터로 판단하도록 학습된다.
    """
    
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=1024, kernel_size=(4, 4), stride=1, padding=0),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=(4, 4), stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z):
        
        return self.network(z)
    
if __name__ == "__main__":
    from torchsummary import torchsummary
    G = Generator().cuda()
    torchsummary.summary(G, input_size = (100, 1, 1))