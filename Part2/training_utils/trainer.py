import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.train_data_loader = kwargs['train_data_loader']
        self.valid_data_loader = kwargs['valid_data_loader']
        self.optimizer = kwargs['optimizer']
        self.loss_fn = kwargs['loss_fn']
        self.args = kwargs['args']
        self.best_acc = float('-inf')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self):
        epochs = self.args.epochs
        for epoch in range(epochs):
            print('start epoch', str(epoch))
            tk = tqdm(self.train_data_loader)
            self.model.train()
            for images, targets in tk:
                images = images.to(self.device)
                targets = targets[:, 0].to(self.device)
                pred = self.model(images)
                loss = self.loss_fn(pred, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tk.set_postfix(train_loss=loss.item())
            tk.close()

            self.eval(epoch)

    def eval(self, epoch):
        self.model.eval();
        acc = 0
        with torch.no_grad():
            tk = tqdm(self.valid_data_loader)
            c = 0 
            correct_sum = 0
            for images, targets in tk:
                images = images.to(self.device)
                targets = targets[:, 0]
                val_output = self.model(images)
                val_output = val_output.cpu()
                val_output = np.argmax(val_output, axis=1)
                correct_sum += torch.sum(val_output == targets)
                c += len(targets)
            acc = correct_sum * 1.0 / c
            acc = acc.item() 
            if acc > self.best_acc:
                self.best_acc = acc
                print('saving the best model, Acc = ', str(acc))
                self.export_model(epoch)
            tk.close()

    def export_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.args.save_path)