import torch.nn as nn

class Discriminator(nn.Module):
    """
    입력 X는 64x64x3 의 이미지 이다. 
    이 모듈은 해당 이미지가 학습데이터 같은지 아닌지 판단한다.
    """
    
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4, 4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(4, 4), stride=1, padding=0),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        return self.network(x)
    
if __name__ == "__main__":
    from torchsummary import torchsummary
    D = Discriminator().cuda()
    torchsummary.summary(D, input_size = (3, 64, 64))