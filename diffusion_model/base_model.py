import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        
    
    @abstractmethod
    def train_step(self, data):
        pass
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for data in dataloader:
            loss = self.train_step(data)
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
