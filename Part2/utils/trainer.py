import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

class Trainer:
    
    def __init__(self, **kwargs):
        self.model = kwargs['model']
        self.train_data_loader = kwargs['train_data_loader']
        self.valid_data_loader = kwargs['valid_data_loader']
        self.optimizer = kwargs['optimizer']
        self.loss_fn = kwargs['loss_fn']
        self.args = kwargs['args']
        self.device = kwargs['device']
        self.writer = kwargs['writer']
        self.scheduler = kwargs['scheduler']
        self.best_acc = float('-inf')

    def train(self):
        epochs = self.args.epochs
        counter = 0
        for epoch in range(epochs):
            print('start epoch', str(epoch))
            tk = tqdm(self.train_data_loader)
            self.model.train()
            for images, targets in tk:
                images = images.to(self.device).float() 
                targets = targets[:, 0].to(self.device)
                pred = self.model(images)
                loss = self.loss_fn(pred, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tk.set_postfix(train_loss=loss.item())
                self.write_log("Train/Loss", loss.item(), counter)
                counter += 1
                # break
            tk.close()
            self.eval(epoch)
            if self.scheduler:
                self.scheduler.step(self.best_acc)

    def eval(self, epoch):
        self.model.eval();
        acc = 0
        with torch.no_grad():
            tk = tqdm(self.valid_data_loader)
            # c = 0 
            # correct_sum = 0
            y_true = []
            y_pred = []
            for images, targets in tk:
                images = images.to(self.device).float() 
                targets = targets[:, 0]
                val_output = self.model(images)
                val_output = val_output.cpu()
                val_output = np.argmax(val_output, axis=1)
                y_pred.extend(val_output)
                y_true.extend(targets)
                # break
                # correct_sum += torch.sum(val_output == targets)
                # c += len(targets)
            # acc = correct_sum * 1.0 / c
            # acc = acc.item() 
            acc = accuracy_score(y_true, y_pred)
            f1_val_macro = f1_score(y_true, y_pred, average='macro')
            f1_val_micro = f1_score(y_true, y_pred, average='micro')
            cm = confusion_matrix(y_true, y_pred)
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm = cm.diagonal()
            self.write_log("Val/Acc", acc, epoch)
            self.write_log("Val/f1_score_macro", f1_val_macro, epoch)
            self.write_log("Val/f1_score_micro", f1_val_micro, epoch)
            for i in range(len(cm)):
                self.write_log("Val/Acc-class " + str(i), cm[i], epoch)
            if acc > self.best_acc:
                self.best_acc = acc
                print('The best model, epoch = ', str(epoch), ', Acc = ', str(acc))
            self.export_model(epoch)
            tk.close()

    def export_model(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, self.args.save_path + "/best_model_epoch_" + str(epoch) + ".pth")


    def write_log(self, key, value, step):
        if self.writer:
            self.writer.add_scalar(key, value, step)
        