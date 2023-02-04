import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import numpy as np
from math import pi
from torch.autograd import Variable
import torch.autograd as autograd
import matplotlib.pyplot as plt

# from layers import SinkhornDistance

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import warnings

warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self, d, N):
        super(Net, self).__init__()
        self.d = d
        self.npoints = N
        self.N = N
        self.eb1 = PairwiseInteractions(d_x=self.d, d_out=32, N=self.N)
        self.eb2 = PairwiseInteractions(d_x=32, d_out=64, N=self.N)
        self.eb3 = PairwiseInteractions_no(d_x=64, d_out=self.d, N=self.N)

    def forward(self, x):
        x0 = x 
        x = self.eb1(x)
        x = self.eb2(x)
        x = self.eb3(x)
        # print("\tIn Model: input size", x0.size(),"output size", x.size())
        torch.cuda.empty_cache()
        return x


class PairwiseInteractions(nn.Module):
    """ Pairwise interactions block taking speeds into account.
    """
    def __init__(self, d_x, d_out, N):
        super(PairwiseInteractions, self).__init__()
        self.npoints = N
        self.d_x = d_x # underlying dimension of input measure
        self.d_out = d_out # underlying dimension of output measure
        self.N = N
        self.meas_x = nn.Linear(2*self.d_x, self.d_out) 

    def forward(self, x0):
        batch_size = x0.size(0)
        # compute pairwise distances for nearest neighbor search.
        distances = torch.sqrt(batch_Lpcost(x0,x0,2,self.d_x))
        # select N nonzero interactions of interest per point.
        val, idx = torch.topk(distances,self.N,2,largest=False,sorted=True)
        distances = None
        val = None
        # tensorized measure of size (batch_size,(N-1)*npoints,2*d) and corresponding speeds

        x0 = batch_index_select_NN(x0.view(batch_size,self.npoints,self.d_x),idx)
        x = x0.clone() # ❗️
        v = None


        # batch multiplication with weights to create new measure x_new.
        x = self.meas_x(x)
        x = F.relu(x)
         # sum over neighbors to create new measure of size (batch_size,npoints,d_out)
        x = x.view(batch_size,self.npoints,self.N-1,self.d_out)
        x = torch.sum(x,2).view(batch_size,self.npoints*self.d_out)
        x /= (self.N-1)
        torch.cuda.empty_cache()
        return x


class PairwiseInteractions_no(nn.Module):
    """ Pairwise interactions block taking speeds into account.
    """
    def __init__(self, d_x, d_out, N):
        super(PairwiseInteractions_no, self).__init__()
        self.npoints = N
        self.d_x = d_x # underlying dimension of input measure
        # self.d_v = d_v # for input speeds
        self.d_out = d_out # underlying dimension of output measure
        self.N = N
        # self.meas_x = nn.Linear(2*self.d_x+2*self.d_v, self.d_out) 
        self.meas_x = nn.Linear(2*self.d_x, self.d_out)


    def forward(self, x0):
        batch_size = x0.size(0)
        # compute pairwise distances for nearest neighbor search.
        distances = torch.sqrt(batch_Lpcost(x0,x0,2,self.d_x))
        # select N nonzero interactions of interest per point.
        val, idx = torch.topk(distances,self.N,2,largest=False,sorted=True)
        distances = None
        val = None
        # tensorized measure of size (batch_size,(N-1)*npoints,2*d) and corresponding speeds

        x0 = batch_index_select_NN(x0.view(batch_size,self.npoints,self.d_x),idx)
        # v = batch_index_select_NN(v0.view(batch_size,self.npoints,self.d_v),idx)
        # x = torch.cat((x0,v),2)
        x = x0.clone()
        v = None


        # batch multiplication with weights to create new measure x_new.
        x = self.meas_x(x)
        # x = F.relu(x)
         # sum over neighbors to create new measure of size (batch_size,npoints,d_out)
        x = x.view(batch_size,self.npoints,self.N-1,self.d_out)
        x = torch.sum(x,2).view(batch_size,self.npoints*self.d_out)
        x /= (self.N-1)
        torch.cuda.empty_cache()
        return x 



def train(epoch, model, args, train_loader, device_idx, optimizer):
    model.train()
    device = device_idx[0]
    # sinkhorn = SinkhornDistance_GPU(eps=5, max_iter=200)
    sinkhorn = SinkhornDistance_GPU(eps=0.1, max_iter=200)
    sinkhorn = nn.DataParallel(sinkhorn, device_ids= device_idx)
    sinkhorn.to(device)
    mean_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.dim = (16,4000)
        # target.dim = (16, 4000)
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        

        output = model(data)

        torch.cuda.empty_cache()

        tmp1 = output.view(output.shape[0], args.npoints, -1)
        tmp2 = target.view(output.shape[0], args.npoints, -1)

        dist1, P, C = sinkhorn(tmp1, tmp2)
        dist2, P, C = sinkhorn(tmp1, tmp1.detach()) 
        dist3, P, C = sinkhorn(tmp2, tmp2)
        dist = dist1*2-dist2-dist3

        loss = dist.mean() *10
        mean_loss += loss.detach()
        loss.backward()
        optimizer.step()
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), loss.item()))

    mean_loss /= (batch_idx +1)
    print('Train Epoch: {} \t Dist: {:.6f}'.format(epoch, mean_loss))
    return mean_loss




def train_v2(epoch, model, args, train_loader, device_idx, optimizer,scheduler):
    model.train()
    device = device_idx[0]
    # sinkhorn = SinkhornDistance_GPU(eps=5, max_iter=200)
    sinkhorn = SinkhornDistance_GPU(eps=0.1, max_iter=200)
    sinkhorn = nn.DataParallel(sinkhorn, device_ids= device_idx)
    sinkhorn.to(device)
    mean_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.dim = (16,4000)
        # target.dim = (16, 4000)
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        
        output = model(data*10) 
        torch.cuda.empty_cache()

        tmp1 = output.view(output.shape[0], args.npoints, -1)
        tmp2 = target.view(output.shape[0], args.npoints, -1)*10 
  

        dist1, P, C = sinkhorn(tmp1, tmp2)   
        dist2, P, C = sinkhorn(tmp1, tmp1.detach())  
        dist3, P, C = sinkhorn(tmp2, tmp2)
        dist = dist1*2-dist2-dist3

        loss = dist.mean()
        mean_loss += loss.detach()
        loss.backward()
        optimizer.step()
        print('finished optimize')

        # scheduler.step() 
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

    mean_loss /= (batch_idx +1)
    print('Train Epoch: {} \t Dist: {:.6f}'.format(epoch, mean_loss))
    return mean_loss



def train_v3(epoch, model, args, train_loader, device_idx, optimizer):
    model.train()
    device = device_idx[0]
    # sinkhorn = SinkhornDistance_GPU(eps=5, max_iter=200)
    sinkhorn = SinkhornDistance_GPU(eps=0.1, max_iter=200)
    sinkhorn = nn.DataParallel(sinkhorn, device_ids= device_idx)
    sinkhorn.to(device)
    mean_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data.dim = (16,4000)
        # target.dim = (16, 4000)
        data, target = data.to(device), target.to(device) 
        optimizer.zero_grad()
        


        output = model(data*10)

        torch.cuda.empty_cache()

       
        tmp1 = output.view(-1, 4)
        tmp2 = target.view(-1, 4)*10 


        dist1, P, C = sinkhorn(tmp1, tmp2)
        dist2, P, C = sinkhorn(tmp1, tmp1.detach()) 
        dist3, P, C = sinkhorn(tmp2, tmp2)
        dist = dist1*2-dist2-dist3

        loss = dist.mean()
        mean_loss += loss.detach()
        loss.backward()
        optimizer.step()
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader), loss.item()))

    mean_loss /= (batch_idx +1)
    print('Train Epoch: {} \t Dist: {:.6f}'.format(epoch, mean_loss))
    return mean_loss




def batch_Lpcost(t1,t2,p,d):
    """ Yields pairwise cost matrix d(t1_i,t2_j)**p for each element of batch.
    """
    batch_size, n_input = t1.size()
    batch_size, m_input = t2.size()
    n,m = int(n_input/d), int(m_input/d)
    t1 = t1.contiguous().view(batch_size,n,d)
    t2 = t2.contiguous().view(batch_size,m,d)
    return torch.sum((torch.abs(t1.unsqueeze(3)-t2.transpose(1,2).unsqueeze(1)))**p,2)
def batch_index_select_NN(x,idx):
    """ Agglomerates initial points and their selected nearest neighbors
        (ending up in dimension 2*d).
        Inputs:
        x: has shape (batch_size,n,d), initial point cloud;
        idx: has shape(batch_size,n,N), nearest neighbors of each point
        from cloud.
        Outputs: tensorized measure of size (batch_size,(N-1)*n,2*d), which
        represents pairwise interactions between points and each of their
        neighbors.
    """
    batch_size, n, d = x.size()
    N = idx.size(2)
    x_pairs = batch_index_select_nNpercloud(x,idx[:,:,1:])
    return torch.cat([x.repeat(1,1,N-1).view(batch_size,(N-1)*n,d), x_pairs], 2).view(batch_size,(N-1)*n,2*d)

def batch_index_select_nNpercloud(x,idx):
    """ x has shape (batch_size,n,d) and idx size (batch_size,n,N).
        Selects N points at indexes idx[i,j] from point j of x[i]-th cloud.
        Ending up with n*N points per cloud, i.e. size (batch_size,N*n,d).
    """
    batch_size, n, d = x.size()
    N = idx.size(2)
    return torch.gather(x.unsqueeze(2).repeat(1,1,N,1),1,idx.unsqueeze(3).repeat(1,1,1,d)).view(batch_size,n*N,d)





def judge_output(pred, xx, epoch, dist, is_x=False):


    if is_x:
        label = 'x_{}'.format(epoch-5) 
        color = 'C2'
    else:
        label = 'y_{}'.format(epoch)
        color = 'C1'


    plt.figure(figsize=(10,7))
    plt.suptitle('dist_true_y ={:.4f}'.format(dist))
    plt.subplot(221)
    t = plt.hist(pred[:,0],bins=100,label='pred_y_{}'.format(epoch))
    t = plt.hist(xx[:,0],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()

    plt.subplot(222)
    t = plt.hist(pred[:,1],bins=100,label='pred_y_{}'.format(epoch))
    t = plt.hist(xx[:,1],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()


    plt.subplot(223)
    t = plt.hist(pred[:,2],bins=100,label='pred_y_{}'.format(epoch))
    t = plt.hist(xx[:,2],bins=100,label=label, color = color, alpha =0.5)

    plt.legend()
    plt.subplot(224)
    t = plt.hist(pred[:,3],bins=100,label='pred_y_{}'.format(epoch))
    t = plt.hist(xx[:,3],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()




def judge_output2(xx, y, epoch, dist):



    label = 'true_y_{}'.format(epoch)
    color = 'C1'


    plt.figure(figsize=(10,7))
    plt.suptitle('dist_x_vs_true_y ={:.4f}'.format(dist))
    plt.subplot(221)
    t = plt.hist(xx[:,0],bins=100,label='x_{}'.format(epoch),color='C2')
    t = plt.hist(y[:,0],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()

    plt.subplot(222)
    t = plt.hist(xx[:,1],bins=100,label='x_{}'.format(epoch),color='C2')
    t = plt.hist(y[:,1],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()


    plt.subplot(223)
    t = plt.hist(xx[:,2],bins=100,label='x_{}'.format(epoch),color='C2')
    t = plt.hist(y[:,2],bins=100,label=label, color = color, alpha =0.5)

    plt.legend()
    plt.subplot(224)
    t = plt.hist(xx[:,3],bins=100,label='true_y_{}'.format(epoch),color='C2')
    t = plt.hist(y[:,3],bins=100,label=label, color = color, alpha =0.5)
    plt.legend()






def pred(model, x):
    output = model(torch.Tensor(x.reshape(1,-1)))
    output_plot = output.detach().cpu().numpy().reshape(-1,4)
    return output_plot



def test(model, x, y, epoch=None, test_path=None):
    pred_y = pred(model, x)
    dist = compute_Was(pred_y, y)
    judge_output(pred_y, y, 50,dist, is_x=False)
    plt.savefig(os.path.join(test_path, 'pred_y_vs_true_y_epoch_{}.pdf'.format(epoch)))
    plt.close()
    dist_xy = compute_Was(x, y)
    judge_output2(x, y, 50, dist_xy)
    plt.savefig(os.path.join(test_path, 'x_vs_y_epoch_{}.pdf'.format(epoch)))
    plt.close()
    return dist,dist_xy


def test2(model, x, y, epoch=None,test_path=None):
    pred_y = pred(model, x*10)
    dist = compute_Was(pred_y/10, y)
    judge_output(pred_y/10, y, 50,dist,  is_x=False)
    plt.savefig(os.path.join(test_path, 'vs_true_y_epoch_{}.pdf'.format(epoch)))
    plt.close()
    judge_output(pred_y/10, x, 50,dist,  is_x=True)
    plt.savefig(os.path.join(test_path, 'vs_x_epoch_{}.pdf'.format(epoch)))
    plt.close()
    return dist




def test_pure(model, x, y, epoch=None,test_path=None):
    pred_y = pred(model, x*10)
    dist = compute_Was(pred_y/10, y)

    return dist





# x.shape=(200,31,4)
def BCO_test(x, num_bco):
  inputs = x[:,:-1,:] # (200,30,4)
  targets = x[:,1:,:] # (200,30,4)

  inputs2 = inputs.reshape(-1,4) 
  targets2 = targets.reshape(-1,4) 

  bco = BCO_cartpole(4, 2, num_bco = num_bco)
  mean_reward = run_BCO(bco, inputs2, targets2)
  return mean_reward



def BCO_test_v2(inputs, targets, num_bco):
    bco = BCO_cartpole(4, 2, num_bco = num_bco)
    mean_reward = run_BCO(bco, inputs, targets)
    return mean_reward


def run_BCO(bco, inputs, targets):
  mean_reward = bco.run_v2(inputs, targets)
  return mean_reward




def test_v3(model, x, y, epoch=None, test_path=None):
  test_true_x_gpu = torch.tensor(test_true_x.reshape(200,-1), dtype=torch.float32).to(device)
  output = model(test_true_x_gpu*10)/10

  pred_y = output.detach().cpu().numpy().reshape(output.shape[0],-1,4)

  pred_y_inputs = pred_y[:,:-1,:] #(200,30,4)
  pred_y_targets = pred_y[:,1:,:] # (200,30,4)

  pred_y_inputs2 = pred_y_inputs.reshape(-1,4) 
  pred_y_targets2 = pred_y_targets.reshape(-1,4) 


  mean_reward = BCO_test_v2(inputs=pred_y_inputs2, targets=pred_y_targets2, num_bco=epoch)
  print('test_epoch:{}, mean_reward_{}'.format(epoch, mean_reward))




  dist1 = compute_Was(x*10, y*10)
  dist2 = compute_Was(pred_y*10, y*10)


  return mean_reward, dist1, dist2








class SinkhornDistance_GPU(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance_GPU, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

        self.test_tensor = nn.Parameter(torch.ones(5,3))


    def forward(self, x, y):

        device = self.test_tensor.device
        # with torch.no_grad():
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
         # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(device)

        u = torch.zeros_like(mu).to(device)
        v = torch.zeros_like(nu).to(device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        self.err = err
        self.actual_iters = actual_nits

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        
        # print('err:{:.4f}, iters:{}'.format(err, actual_nits))
        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1



def dict_to_object(dic):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    return Struct(**dic)




def compute_Was(d1,d2,device=torch.device("cuda: 0")):
    tmp1 = torch.Tensor(d1).to(device)
    tmp2 = torch.Tensor(d2).to(device)
    sinkhorn = SinkhornDistance_GPU(eps=0.1, max_iter=200)
    sinkhorn.to(device)
    dist1, P, C = sinkhorn(tmp1, tmp2)
    dist2, P, C = sinkhorn(tmp2, tmp2)
    dist3, P, C = sinkhorn(tmp1, tmp1)
    dist = dist1*2-dist2-dist3
    return dist




class FlockingCloudDataset(data.Dataset):
    """ Cucker-Smale flocking dataset.
    """
    def __init__(self, d, npoints, root, train = False, scale_x = None, scale_y = None):
        self.d = d
        self.npoints = npoints
        self.root = root
        self.train = train
        # self.scale_x = scale_x        # self.scale_y = scale_y

        self.scale_x = 1.0
        self.scale_y = 1.0

        xy = np.loadtxt(self.root, delimiter = ',') 
        # xy.shape = (80,8000)
        if(len(xy.shape) == 1):
            xy = xy.reshape(1, -1)
        


        # xy = self.normalize(xy)

        self.x_data = torch.from_numpy(xy[:50,:23000]) # first (measures,velocities)
        self.y_data = torch.from_numpy(xy[:50,23000:]) # true final measures
        self.len = self.x_data.shape[0]


    def __getitem__(self,index):
        mu_0 = self.x_data[index,:self.d*self.npoints].unsqueeze(0).float()
        v_0 = self.x_data[index,self.d*self.npoints:].unsqueeze(0).float()
        mu_2 = self.y_data[index]
        return torch.cat((mu_0,v_0),1).squeeze(0), mu_2
    def __len__(self):
        return self.len



    def normalize(self, xy):
        x = np.concatenate([xy[:,:1500],xy[:, 3000:]], 1)
        y = xy[:,1500:3000] 


        if(self.train == True):
            self.scale_x = max(abs(x.max()), abs(x.min()))
            self.scale_y = max(abs(y.max()), abs(y.min()))
            
        norm_x = x/self.scale_x
        norm_y = y/self.scale_y

        norm_xy = np.concatenate([norm_x[:,:1500], norm_y, norm_x[:,1500:]], 1)
        return norm_xy
