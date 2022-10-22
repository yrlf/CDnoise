from torch.nn import Module
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
__all__ = ["NaiveNet", "NaiveNet2","sig_t","NaiveNet3","LeNet5"]



class LeNet5(nn.Module):

    def __init__(self,feature_dim=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(400, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.l2norm = Normalize(2)
        
        #projection MLP
        self.fc4 = nn.Linear(400, 800)
        self.fc5 = nn.Linear(800, 40)


    def forward(self, x):

        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        feat = out.view(out.size(0), -1)
     
        out = F.relu(self.fc1(feat))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)


        return out



class NaiveNet(nn.Module):

    def __init__(self, feature_dim=2, hidden_dim=25, num_classes=2, pretrained=False, input_channel=1):
        super(NaiveNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out=F.dropout(out, p=0.25)
        out = F.relu(self.fc2(out))
        out=F.dropout(out, p=0.25)
        out = self.fc3(out)
        return out

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class NaiveNet3(nn.Module):

    def __init__(self, feature_dim=2, hidden_dim=25, num_classes=2, pretrained=False, input_channel=1):
        super(NaiveNet3, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        self.l2norm = Normalize(2)
        
        #projection MLP
        self.fc4 = nn.Linear(hidden_dim, 10)
        self.fc5 = nn.Linear(10, 5)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out=F.dropout(out, p=0.25)
        out = F.relu(self.fc2(out))
        feat=F.dropout(out, p=0.25)
        out = self.fc3(feat)
        feat = F.relu(self.fc4(feat))
        feat = self.fc5(feat)
        feat = self.l2norm(feat)

        return out, feat



class NaiveNet2(nn.Module):

    def __init__(self, feature_dim=2, hidden_dim=25, num_classes=2, pretrained=False, input_channel=1):
        super(NaiveNet2, self).__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
        # self.T_revision = nn.Linear(num_classes, num_classes, False)

    def forward(self, x, revision=True):
        correction = None
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        if revision == True:
            return out, correction
        else:
            return out, correction
        


class sig_t(nn.Module):
    def __init__(self, device, num_classes, init=2):
        super(sig_t, self).__init__()

        self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(num_classes, num_classes)))

        self.w.cuda()

        co = torch.ones(num_classes, num_classes)
        ind = np.diag_indices(co.shape[0])
        co[ind[0], ind[1]] = torch.zeros(co.shape[0])
        self.co = co.cuda()
        self.identity = torch.eye(num_classes).cuda()


    def forward(self):
        sig = torch.sigmoid(self.w)
        T = self.identity.detach() + sig*self.co.detach()
        T = F.normalize(T, p=1, dim=1)

        return T