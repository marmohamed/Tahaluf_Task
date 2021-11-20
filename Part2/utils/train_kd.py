import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import os

class KDTrainer:
    
    def __init__(self, gpu=None, rank=None, **kwargs):
        self.gpu = gpu
        self.rank = rank
        self.model = kwargs['model']
        self.train_data_loader = kwargs['train_data_loader']
        self.valid_data_loader = kwargs['valid_data_loader']
        self.optimizer = kwargs['optimizer']
        self.loss_fn = kwargs['loss_fn']
        self.args = kwargs['args']
        self.device = kwargs['device']
        self.writer = kwargs['writer']
        self.scheduler = kwargs['scheduler']
        self.model_student = kwargs['model_student']
        self.mixed_precision = self.args.mixed_precision
        self.best_acc = float('-inf')

    def train(self):
        epochs = self.args.epochs
        counter = 0
        if self.mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(epochs):
            print('start epoch', str(epoch))
            tk = tqdm(self.train_data_loader)
            self.model.eval()
            self.model_student.train()
            for images, targets in tk:
                if self.args.train_multinode and self.gpu is not None:
                    images = images.cuda(non_blocking=True).float()
                    targets = targets[:, 0].cuda(non_blocking=True)
                else:
                    images = images.to(self.device).float() 
                    targets = targets[:, 0].to(self.device)
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        pred = self.model_student(images)
                        # loss = self.loss_fn(pred, targets)
                else:
                    pred = self.model_student(images)
                    # loss = self.loss_fn(pred, targets)

                with torch.no_grad():
                    output_teacher_batch = self.model(images).to(self.device)

                alpha = 0.2
                T = 2
                l1 = torch.nn.KLDivLoss()(torch.nn.functional.log_softmax(pred/T, dim=1),
                                        torch.nn.functional.softmax(output_teacher_batch/T, dim=1)) * (alpha * T * T)
                l2 = self.loss_fn(pred, targets) * (1. - alpha)
                loss = l1 + l2


                self.optimizer.zero_grad()
                if self.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                tk.set_postfix(train_loss=loss.item())
                self.write_log("Train/Loss", loss.item(), counter)
                counter += 1
                break
            tk.close()
            self.eval(epoch)
            if self.scheduler:
                self.scheduler.step(self.best_acc)

    def eval(self, epoch):
        self.model.eval()
        self.model_student.eval()
        # acc = 0
        with torch.no_grad():
            tk = tqdm(self.valid_data_loader)
            # c = 0 
            # correct_sum = 0
            y_true = []
            y_pred = []
            for images, targets in tk:
                if self.args.train_multinode and self.gpu is not None:
                    images = images.cuda(non_blocking=True).float()
                    # targets = targets[:, 0].cuda(non_blocking=True)
                else:
                    images = images.to(self.device).float() 
                    # targets = targets[:, 0].to(self.device)
                # images = images.to(self.device).float() 
                targets = targets[:, 0]
                val_output = self.model_student(images)
                val_output = val_output.cpu()
                val_output = np.argmax(val_output, axis=1)
                y_pred.extend(val_output)
                y_true.extend(targets)
                break
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
        if (self.args.train_multinode and self.rank == 0) or not self.args.train_multinode:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model_student.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.args.save_path, self.args.experiment_name, "best_model_epoch_" + str(epoch) + ".pth"))


    def write_log(self, key, value, step):
        if self.writer:
            self.writer.add_scalar(key, value, step)
        