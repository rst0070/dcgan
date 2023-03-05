from torch.utils.data import DataLoader
import Generator
import Discriminator
import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    
    def __init__(self, dataset, generator:Generator, discriminator:Discriminator):
        
        self.batch_size = 50
        self.dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.generator = generator
        self.discriminator = discriminator
        
        self.loss_fn = nn.BCELoss().cuda()
        
        self.lr = 0.001
        self.optG = torch.optim.Adam(
            params = self.generator.parameters(),
            lr = self.lr
        )
        self.optD = torch.optim.Adam(
            params = self.discriminator.parameters(),
            lr = self.lr
        )
        
        
        
    def train(self):
        
        pbar = tqdm(self.dataloader)
        itered = 0
        fake_list = []
        
        for X, y in pbar:
            
            itered += 1
            batch_size = X.shape[0]
            label_zero = torch.zeros((batch_size, 1, 1, 1), device="cuda")
            label_one = torch.ones((batch_size, 1, 1, 1), device="cuda")
                
            """
            training D
            """
            self.discriminator.zero_grad()
            
            #print(X)
            # real data
            pred = self.discriminator(X.cuda())
            loss = self.loss_fn(pred, label_one)
            loss.backward(retain_graph = True)
            # fake data
            z = torch.randn(batch_size, 100, 1, 1, device="cuda")
            fake_x = self.generator(z)
            pred = self.discriminator(fake_x)
            loss = self.loss_fn(pred, label_zero)
            loss.backward(retain_graph = True)
            # 
            self.optD.step()
            
            """
            training G
            """
            self.generator.zero_grad()
            pred = self.discriminator(fake_x)
            loss = self.loss_fn(pred, label_one)
            loss.backward(retain_graph = True)
            self.optG.step()
            
            if itered % 500 == 0:
                fake_list.append(fake_x.cpu())
                
        return fake_list