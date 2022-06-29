import sys
import shutil
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np



# Residual block
class ResBlock(nn.Module):
    def __init__(self, dim=512):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(dim,dim)
        self.fc2 = nn.Linear(dim,dim)# has bias
        self.bias = nn.Parameter(torch.ones(dim))
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        out = out + x +self.bias
        out = self.dropout(out)
        return F.relu(out)
    
# TDNN block
class TDNN(nn.Module):
    def __init__(self, context, input_dim=512, output_dim=512, full_context = False):
        """
        https://github.com/SiddGururani/Pytorch-TDNN
        """
        super(TDNN,self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.check_valid_context(context)
        self.kernel_width, context = self.get_kernel_width(context,full_context)
        self.register_buffer('context',torch.LongTensor(context))
        self.full_context = full_context
        stdv = 1./math.sqrt(input_dim)
        self.kernel = nn.Parameter(torch.Tensor(output_dim, input_dim, self.kernel_width).normal_(0,stdv))
        self.bias = nn.Parameter(torch.Tensor(output_dim).normal_(0,stdv))

    def forward(self,x):
        conv_out = self.special_convolution(x, self.kernel, self.context, self.bias)
        return conv_out

    def special_convolution(self, x, kernel, context, bias):
        input_size = x.size()
        assert len(input_size) == 3, 'Input tensor dimensionality is incorrect. Should be a 3D tensor'
        [batch_size, input_sequence_length, input_dim] = input_size
        x = x.transpose(1,2).contiguous()

        # Allocate memory for output
        valid_steps = self.get_valid_steps(self.context, input_sequence_length)
        xs = Variable(self.bias.data.new(batch_size, kernel.size()[0], len(valid_steps)))

        # Perform the convolution with relevant input frames
        for c, i in enumerate(valid_steps):
            features = torch.index_select(x, 2, context+i)
            xs[:,:,c] = F.conv1d(features, kernel, bias = bias)[:,:,0]
        return xs.transpose(1,2).contiguous()
    
    @staticmethod
    def check_valid_context(context):
        # here context is still a list
        assert context[0] <= context[-1], 'Input tensor dimensionality is incorrect. Should be a 3D tensor'

    @staticmethod
    def get_kernel_width(context, full_context):
        if full_context:
            context = range(context[0],context[-1]+1)
        return len(context), context

    @staticmethod
    def get_valid_steps(context, input_sequence_length):
        start = 0 if context[0] >= 0 else -1*context[0]
        end = input_sequence_length if context[-1] <= 0 else input_sequence_length - context[-1]
        return range(start, end)
    
    
    
    
class SelfAttentiveLayer(nn.Module):
    def __init__(self, atten_in_dim,attention_unit,num_heads=5):
        super(SelfAttentiveLayer, self).__init__()
        self.ws1 = nn.Linear(atten_in_dim, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, num_heads, bias=False)
        self.tanh = nn.Tanh()
        self.init_weights()
        self.attention_hops = num_heads
        self.dropout = nn.Dropout(p=0.5)

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, x):
        # [bsz, len, nhid]
        # 200,100,128
        H=x
        A = self.tanh(self.ws1(x))    #200*100*128 * 128*128 = 200*100*128
        A = self.ws2(A)               #200*100*128 * 128*5 = 200*100*5
        A = self.softmax(A,1)         #column-wise softmax
        A = A.transpose(1,2)          #200,5,100
        output = torch.bmm(A, H)      #200*(5*100 * 100*128)=200,5,128
        output = self.dropout(output)
        
        return output , A   
    
    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d,dim=1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
    
    
class Time_sync(nn.Module):
    def __init__(self, input_dim=500, hidden_dim=512, atten_in_dim=128,attention_unit=128,num_heads=5):
        super(Time_sync,self).__init__()
        self.fc1=nn.Linear(input_dim,hidden_dim)
        self.ResLayer1 = ResBlock(hidden_dim)
        self.TDNN2=TDNN([-2,1],hidden_dim,hidden_dim, full_context=False)
        self.ResLayer2 = ResBlock(hidden_dim)
        self.TDNN3=TDNN([-3,3],hidden_dim,hidden_dim, full_context=False)
        self.ResLayer3 = ResBlock(hidden_dim)
        self.TDNN4=TDNN([-7,2],hidden_dim,hidden_dim, full_context=False)
        self.ResLayer4 = ResBlock(hidden_dim)
        self.bn_frame = nn.Linear(hidden_dim,atten_in_dim)
        self.Atten = SelfAttentiveLayer(atten_in_dim,attention_unit,num_heads)
        self.bn_window = nn.Linear(attention_unit*num_heads,128)

        
    def forward(self,x):
        #[batch_size, sequence_length, input_dim]
        # 200,118,500
        x=F.relu(self.fc1(x))
        x=self.ResLayer1(x)
        # 200, 118, 512
        x=F.relu(self.TDNN2(x))
        x=self.ResLayer2(x)
        # 200, 115, 512
        x=F.relu(self.TDNN3(x))
        x=self.ResLayer3(x)
        # 200, 109, 512
        x=F.relu(self.TDNN4(x))
        x=self.ResLayer4(x)
        # 200, 100, 512
        x=self.bn_frame(x)
        # 200, 100, 128
        x, A = self.Atten(x)
        # x: 200, 5, 128
        # A: 200, 5, 100
        x=F.relu(self.bn_window(x.view(x.size(0),-1).contiguous()))
        # 200, 128
        return x , A
    
class Time_async(nn.Module):
    def __init__(self, bert_dim=768, atten_in_dim=64,attention_unit=64,num_heads=5):
        super(Time_async,self).__init__()
        self.layer_bert =  nn.Linear(bert_dim,atten_in_dim)
        self.bert_atten =  SelfAttentiveLayer(atten_in_dim,attention_unit,num_heads)
        
    def forward(self,x):
        size=x.size()  # 200,768*7
        bert=[]
        for i in range(7):
            bert.append(F.relu(self.layer_bert(x[:,768*i:768*(i+1)])).unsqueeze(1))  #200,768 --> 200,1,64
        bert_all = torch.cat((bert),1)  # 200,7,64
        out, A = self.bert_atten(bert_all)  # out: 200, 5, 64  A: 200, 5, 7
        return out.view(out.size(0),-1).contiguous(), A # out: 200, 320  A: 200, 5, 7
    
    
class Fusion(nn.Module):
    def __init__(self,fuse_dim=128, output_dim=5):
        super(Fusion,self).__init__()
        self.b1 = Time_sync()
        self.b2 = Time_async()
        self.layer_cat = nn.Linear(128+64*5,fuse_dim)
        self.layer_out = nn.Linear(fuse_dim,output_dim,bias=False)
        
    def forward(self,x,y):
        # x: 200,118,500  y: 200, 5376
        x,A1 = self.b1(x) # x: 200,128
        y,A2 = self.b2(y) # y: 200,320
        out = torch.cat((x,y),1)  #200,448
        out = F.relu(self.layer_cat(out))  #200,128
        out = self.layer_out(out)  #200,4
        return out,A1,A2

    

class newbob():
    def __init__(self, optimizer, factor, model):
        self.optimizer = optimizer
        self.factor = factor
        self.model = model
        self.Flast = 0.0
        self.model_epoch = 0
        self.ramp = 0
        self.N = 0
        self.Nmax = 20
        self.threshold = 0.001
        self.model_state = None
        self.optimizer_state = None
        self.lr = get_lr(optimizer)
        self.initlr = 0 

    def step(self, F):
        self.N += 1
        print("N:",self.N)
        dF = F - self.Flast
        print(F,self.Flast,dF)
        if dF <= 0:
            print("loading model state from epoch:",self.model_epoch)
            self.model.load_state_dict(self.model_state)
            self.optimizer.load_state_dict(self.optimizer_state)
        else:
            print("saving model state")
            self.Flast = F
            self.model_epoch = self.N
            self.model_state = self.model.state_dict()
            self.optimizer_state = self.optimizer.state_dict()
        print("ramp:",self.ramp)

        if self.ramp:
            self.lr = self.lr /2
        elif dF < self.threshold:
            self.lr = self.lr /2
            if self.N >= self.Nmax:
                self.ramp=True

        print("lr:",self.lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

