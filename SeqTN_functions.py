import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as data

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics


def train(args,train_loader, model):
    '''
    args: dictionary with key 'use_cuda'=True if using GPUs
    train_loader: data loader for training data
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['use_cuda']:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
            
def test(args, test_loader, model):
    '''
    args: dictionary with key 'use_cuda'=True if using GPUs
    test_loader: data loader for test data
    '''   
    model.eval()
    test_loss = 0
    correct = 0
    predicted=[]
    y_true=[]
    
    for data, target in test_loader:
        if args['use_cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if use_cuda:
            output=output.cpu()
            target=target.cpu()
        predicted=predicted+list(output[:,1].data.numpy().flatten())
        y_true=y_true+list(target.data.numpy().flatten())
        
    test_loss /= len(test_loader.dataset)
    auc=roc_auc_score(y_true,predicted)
    aupr=average_precision_score(y_true,predicted)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), AUC: {:.4f}, AUPR: {:.4f}\n'
      .format(test_loss, correct, len(test_loader.dataset),
              100. * correct / len(test_loader.dataset),auc,aupr))
    
    return auc,aupr


def HP_Search(args,train_loader,val_loader,model):
    
    '''
    Hyperparameter Search
    - train_loader/val_loader: data loader for training and validation set
    - model: model to HP tune
    - args: dictionary with values for
        'budget': Search budget
        'epochs': Max # epochs to train each model
        'use_cuda': True if using GPUs
        'savepath': save directory
    '''
    df=pd.DataFrame({'auc':[],'batch_size':[],'dropout':[],'depth':[],'num_neurons':[]})
    auc_all=0

    for run in range(args['budget']):
        print('******Run: {}'.format(run))

        #Draw HP 
        args['batch_size']=int(np.random.choice([8,15,30]))
        args['dropout']=np.random.choice([0,.1,.2,.3,.4,.5,.6,.7,.8,.9])
        args['depth']=int(np.random.choice([2,3,4]))
        args['num_neurons']=int(np.random.choice([50,100,250,500]))

        if args['use_cuda']:
            model.cuda()
        optimizer = optim.Adam(model.parameters()) #optim.SGD(model.parameters(), lr=0.01)

        #Train Model
        auc_run=0
        epoch_run=0
        for epoch in range(1, args['epochs'] + 1):
            train(args,train_loader, model)
            zauc,_=test(args,val_loader,model)
            if zauc>auc_all:
                torch.save(model.state_dict(),args['savepath']+'bestmodel.pth.tar')
                auc_all=zauc
            if zauc>auc_run: 
                auc_run=zauc
                epoch_run=epoch

        df=df.append(pd.DataFrame({'auc':[auc_run],'batch_size':[args['batch_size']],'dropout':[args['dropout']]
                               ,'depth':[args['depth']],'num_neurons':[args['num_neurons']]
                               ,'epoch':[epoch_run]}))
        df.to_pickle(args['savepath']+'HP_search.pickle')
    
    return


class restrictedActivation(nn.Module):
    '''
    Gradient Clipping for Sequence Transformer Networks
    '''
    
    def __init__(self):
        super(restrictedActivation, self).__init__()
        self.threshold=Variable(torch.FloatTensor([2.0]).cuda())
        self.reduction=Variable(torch.FloatTensor([.01]).cuda())
        
    def forward(self, x):
        negmask=(x<-self.threshold).float()
        posmask=(x>self.threshold).float()
        mask=((x>=-self.threshold)&(x<=self.threshold)).float()
        
        newx=torch.add(torch.mul(x,mask), torch.mul(-self.threshold,negmask))
        newx=torch.add(newx, torch.mul(self.threshold,posmask))
        newx=torch.add(newx, torch.mul(torch.mul(x,negmask), self.reduction))
        newx=torch.add(newx, torch.mul(torch.mul(x,posmask), self.reduction) )
        return newx


class SeqTN(nn.Module):
    def __init__(self, args):
        '''
        Sequence Transformer Network
        - args: dictionary with values for
            'use_cuda': True if using GPUs
            'depth': CNN depth (2,3,4)
            'filter_size': filter size of CNN
            'dropout': dropout ratio
            'num_neurons': number of units in fully connected
            'num_classes': label class #
        '''
        
        super(SeqTN, self).__init__()
        self.args=args

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        n_size = self._get_conv_output(mode=1)
        self.fc_loc = nn.Sequential(
            nn.Linear(n_size, 32),
            nn.ReLU(True),
            nn.Linear(32,2+2),
            restrictedActivation()
        )
        
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 1, 0])
        if args['use_cuda']==True: self.restriction = torch.autograd.Variable(torch.FloatTensor([1, 0, 0, 0]).cuda(), requires_grad = False)
        else: self.restriction = torch.autograd.Variable(torch.FloatTensor([1, 0, 0, 0]), requires_grad = False)

        
        if args['depth']==2:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes'])
            
        elif args['depth']==3:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            self.conv3 = nn.Conv1d(20, 40, kernel_size=args['filter_size'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes']) 
            
        elif args['depth']==4:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            self.conv3 = nn.Conv1d(20, 40, kernel_size=args['filter_size'])
            self.conv4 = nn.Conv1d(40, 80, kernel_size=args['filter_size'])
            self.conv4_drop = nn.Dropout(p=args['dropout'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes'])

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.num_flat_features(xs))
        theta = self.fc_loc(xs)
        theta1 = theta[:,:2]
        theta2 = theta[:,2:]
        
        theta1 = torch.cat((self.restriction.repeat(theta1.shape[0],1), theta1), 1)
        theta1 = theta1.view(-1, 2, 3)
        grid = F.affine_grid(theta1, x.size()) 
        x = F.grid_sample(x, grid, padding_mode='border')
        
        thetaw = theta2[:,0].contiguous().view(x.shape[0],1,1,1)
        thetab = theta2[:,1].contiguous().view(x.shape[0],1,1,1)
        x = torch.mul(x, thetaw)
        x = torch.add(x, thetab)
        return x
    
    def forward(self, x):
        x = self.stn(x)

        if self.args['depth']==2:
            x = self.conv1(x)
            x = x.view(x.shape[:-1]) 
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        
        elif self.args['depth']==3:
            x = self.conv1(x)
            x = x.view(x.shape[:-1]) 
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        
        elif self.args['depth']==4:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(x)
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = F.relu(F.max_pool1d(self.conv4_drop(self.conv4(x)), 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    def _get_conv_output(self, mode):
        bs = 1
        input = Variable(torch.rand(bs, 1, self.args['T'], self.args['d']))
        if mode==1: #size for the regressor
            output_feat = self.localization(input)
            n_size = output_feat.data.view(bs, -1).size(1)
        else:
            output_feat = self._forward_features(input, mode)
            n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    def _forward_features(self, x, mode):    
        if mode==2:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        if mode==3:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
        if mode==4:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(x)
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = F.relu(F.max_pool1d(self.conv4_drop(self.conv4(x)), 2))
        return x

class CNN(nn.Module):
    def __init__(self, args):
        '''
        Baseline CNN
        - args: dictionary with values for
            'use_cuda': True if using GPUs
            'depth': CNN depth (2,3,4)
            'filter_size': filter size of CNN
            'dropout': dropout ratio
            'num_neurons': number of units in fully connected
            'num_classes': label class #
        '''
        super(CNN, self).__init__()
        self.args=args

        if args['depth']==2:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes'])
            
        elif args['depth']==3:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            self.conv3 = nn.Conv1d(20, 40, kernel_size=args['filter_size'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes']) 
            
        elif args['depth']==4:
            self.conv1 = nn.Conv2d(1, 10, kernel_size=(args['filter_size'],self.args['d']))
            self.conv2 = nn.Conv1d(10, 20, kernel_size=args['filter_size'])
            self.conv2_drop = nn.Dropout(p=args['dropout'])
            self.conv3 = nn.Conv1d(20, 40, kernel_size=args['filter_size'])
            self.conv4 = nn.Conv1d(40, 80, kernel_size=args['filter_size'])
            self.conv4_drop = nn.Dropout(p=args['dropout'])
            n_size = self._get_conv_output(mode=args['depth'])
            self.fc1 = nn.Linear(n_size, args['num_neurons'])
            self.fc2 = nn.Linear(args['num_neurons'],args['num_classes'])

    def num_flat_features(self, x):
        size = x.size()[1:]  
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        if self.args['depth']==2:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        
        elif self.args['depth']==3:
            x = self.conv1(x)
            x = x.view(x.shape[:-1]) 
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)
        
        elif self.args['depth']==4:
            x = self.conv1(x)
            x = x.view(x.shape[:-1]) 
            x = F.relu(x)
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = F.relu(F.max_pool1d(self.conv4_drop(self.conv4(x)), 2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    def _get_conv_output(self, mode):
        bs = 1
        input = Variable(torch.rand(bs, 1, self.args['T'], self.args['d']))
        output_feat = self._forward_features(input, mode)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size
    
    def _forward_features(self, x, mode):
        if mode==2:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
        if mode==3:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(F.max_pool1d(x, 2))
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
        if mode==4:
            x = self.conv1(x)
            x = x.view(x.shape[:-1])
            x = F.relu(x)
            x = F.relu(F.max_pool1d(self.conv2_drop(self.conv2(x)), 2))
            x = F.relu(self.conv3(x))
            x = F.relu(F.max_pool1d(self.conv4_drop(self.conv4(x)), 2))
        return x       