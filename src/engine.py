import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from dataset import get_loader
import warnings
warnings.simplefilter("ignore", UserWarning)
import sys
sys.path.append('./models/')
from build import build_model


class Trainer:
    def __init__(self, model, criterion, optimizer, ema_model, loss_ratio=0.1,
                 clip_value=1, ckpt='../weights/model.pth', device='cuda'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_ratio = loss_ratio
        self.clip_value = clip_value
        self.device = device
        self.ema_model = ema_model
        self.BEST_LOSS = np.inf
        self.ckpt = ckpt
        self.labels = []
        self.predicts = []
        self.load_weights()

    def load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.ckpt))
            # self.ema_model.store(self.model.parameters())
            # self.ema_model.copy(self.model.parameters())
            print("SUCCESSFULLY LOAD TRAINED MODELS !")
        except:
            print('FIRST TRAINING >>>')

    def train_epoch(self, train_loader):
        self.model.train()
        train_loss_epoch = 0
        train_acc_epoch = []
        for img, label in tqdm(train_loader):
            self.optimizer.zero_grad()

            img = img.to(self.device)
            feature, out = None, self.model(img)

            label = label.to(self.device)
            loss = self.criterion(out, label.squeeze(dim=1))
            train_loss_epoch += loss.item()
            loss.backward()
            
            # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.clip_value)
            self.optimizer.step()
            self.ema_model.update(self.model.parameters())

            _, predict = out.max(dim=1)
            train_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
        return sum(train_acc_epoch) / len(train_acc_epoch), train_loss_epoch

    def val_epoch(self, val_loader):
        self.model.eval()
        self.ema_model.store(self.model.parameters())
        self.ema_model.copy(self.model.parameters())
        val_loss_epoch = 0
        val_acc_epoch = []
        with torch.no_grad():
            for img, label in tqdm(val_loader):
                img = img.to(self.device)
                feature, out = None, self.model(img)
                
                label = label.to(self.device)
                loss = self.criterion(out, label.squeeze(dim=1))
                val_loss_epoch += loss.item()

                _, predict = out.max(dim=1)
                self.labels.append(label.squeeze(dim=1).detach().cpu().numpy())
                self.predicts.append(predict.detach().cpu().numpy())
                val_acc_epoch.append(accuracy_score(predict.cpu().numpy(), label.cpu().numpy()))
                self.val_loss_epoch = val_loss_epoch
                self.ema_model.copy_back(self.model.parameters())
            return sum(val_acc_epoch) / len(val_acc_epoch), val_loss_epoch

    def save_checkpoint(self, experiment):
        if self.val_loss_epoch < self.BEST_LOSS:
            self.BEST_LOSS = self.val_loss_epoch
            torch.save(self.model.state_dict(),self.ckpt)
            experiment.log_model("model",self.ckpt)
            experiment.log_confusion_matrix(y_true=[j for sub in self.labels for j in sub],
                                            y_predicted=[j for sub in self.predicts for j in sub])
            print("LOG CONFUSION MATRIX")


def training_experiment(train_loader, test_loader, experiment, trainer, epoch_n, scheduler):
    print("BEGIN TRAINING ...")
    for epoch in range(epoch_n):
        with experiment.train():
            mean_train_acc, train_loss_epoch = trainer.train_epoch(train_loader)
            experiment.log_metrics({
                "loss": train_loss_epoch,
                "acc": mean_train_acc
            }, epoch=epoch)

        with experiment.test():
            mean_val_acc, val_loss_epoch = trainer.val_epoch(test_loader)
            scheduler.step(val_loss_epoch)
            trainer.save_checkpoint(experiment)
            experiment.log_metrics({
                "loss": val_loss_epoch,
                "acc": mean_val_acc
            }, epoch=epoch)

        print("EPOCH: ", epoch + 1, " - TRAIN_LOSS: ", train_loss_epoch, " - TRAIN_ACC: ", mean_train_acc,
              " || VAL_LOSS: ", val_loss_epoch, " - VAL_ACC: ", mean_val_acc)