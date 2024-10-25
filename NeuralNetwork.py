import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns

class NeuralNetwork(nn.Module):
    def __init__(self, no_input = 24, no_labels = 12, train=False,lr = 0.01, weights = None,prev_epoch=0):
        super(NeuralNetwork, self).__init__()
        self.weight_dir = './weights/'
        self.plot_dir = './loss_plot/'
        self.conf_dir = './conf_plot/'
        self.no_input = no_input
        self.no_hidden1 = 128
        self.no_hidden2 = 64
        self.no_hidden3 = 16
        self.no_out = no_labels
        self.prev_epoch = prev_epoch
        self.train = train
        self.init_model()
        if weights is not None:
            self.weights = weights
            self.model.load_state_dict(torch.load(self.weights))
            print('Previous Weights loaded')
        if train:
            self.learning_rate = lr
            self.loss_function = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        else:
            self.model.eval()


    def init_model(self):
        self.model = nn.Sequential(nn.Linear(self.no_input, self.no_hidden1),
                                   nn.ReLU(),
                                   nn.Linear(self.no_hidden1, self.no_hidden2),
                                   nn.ReLU(),
                                   nn.Linear(self.no_hidden2, self.no_hidden3),
                                   nn.ReLU(),
                                   nn.Linear(self.no_hidden3, self.no_out),
                                   nn.Softmax())
        print(self.model)



    def forward(self, x):
        return self.model(x)

    def save_weights(self,loss = None, epochs=None):
        if epochs is not None and loss is not None:
            file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':'))+'_classes_'+ str(self.no_out)+'_epochs_'+str(epochs+1+self.prev_epoch)+'_loss_'+str(loss)+'.pth'
        elif epochs is not None:
            file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':'))+'_classes_'+ str(self.no_out)+'_epochs_'+str(epochs+1+self.prev_epoch)+'.pth'
        elif loss is not None:
            file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':'))+'_classes_'+ str(self.no_out)+'_loss_'+str(loss)+'.pth'
        else:
            file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':'))+'_classes_'+str(self.no_out) +'.pth'
        torch.save(self.model.state_dict(), self.weight_dir+file_name)
        return self.weight_dir+file_name

    def train_model(self,train_dataloader,epochs):
        losses = []
        for epoch in range(epochs):
            for X, y in train_dataloader:
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                pred = self.model(X)
                loss = self.loss_function(pred, y)
                loss.backward()
                self.optimizer.step()
            print('Epoch:',epoch+1,loss.item())
            if(epoch+1) % 200 == 0:
                losses.append(loss.item())
                self.save_weights(loss =loss.item(), epochs=epoch)

        return losses

    def test_model(self, test_dataloader):
        y_pred = []
        y_test = []
        total = 0
        with torch.no_grad():
            for X, y in test_dataloader:
                pred = self.model(X)
                y_pred.append(pred.detach().numpy())
                y_test.append(y.detach().numpy())
                total += y.size(0)

        return np.array( y_pred), np.array(y_test), total

    def get_confusion_matrix(self,y_test, y_pred,show = True):
        print(classification_report(y_test, y_pred))
        cf_matrix = confusion_matrix(y_test, y_pred)
        plt.subplots(figsize=(8, 5))
        sns.heatmap(cf_matrix, annot=True, cbar=False, fmt="g")
        file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':')) + '.png'
        plt.savefig(self.conf_dir + file_name)
        if show:
            plt.show()
        return cf_matrix



    def plot_losses(self, losses,show = True):
        plt.plot(losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("Learning rate %f" % (self.learning_rate))
        file_name = '-'.join('_'.join(str(datetime.datetime.now()).split('.')[0].split()).split(':')) + '.png'
        plt.savefig(self.plot_dir + file_name)
        if show:
            plt.show()


    def inference(self, data_x):
        pred_y = self.model(data_x)
        return pred_y


