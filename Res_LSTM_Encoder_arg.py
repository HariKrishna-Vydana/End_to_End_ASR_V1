#!/usr/bin/python
import torch.nn as nn
import torch
import torch.nn.functional as F
import math 
import sys


###########################################################
class subsampling_LSTMs(nn.Module):
        def __init__(self, args):
                super(subsampling_LSTMs, self).__init__()
                self.input_size   =   int(args.input_size)
                self.hidden_size  =   int(args.hidden_size)
                self.conv_dropout =   args.conv_dropout

                self.subsampling_LSTM1 = nn.LSTM(self.input_size,self.hidden_size,1,batch_first=False,bidirectional=True,dropout=self.conv_dropout)
                self.PROJ_Layer1 = nn.Linear(self.hidden_size*2, self.hidden_size)
                self.Dropout_layer1 = nn.Dropout(p=self.conv_dropout)

                self.subsampling_LSTM2 = nn.LSTM(self.hidden_size,self.hidden_size,1,batch_first=False,bidirectional=True,dropout=self.conv_dropout)
                self.PROJ_Layer2 = nn.Linear(self.hidden_size*2, self.hidden_size)
                self.Dropout_layer2 = nn.Dropout(p=self.conv_dropout)

        def forward(self, conv_input):
                #print("insideconv_layer")
                #import pdb;pdb.set_trace()
                conv_input=conv_input.transpose(0,1)
                #print(conv_input.shape)
                #-----------------------------------------
                LSTM_output, hidden1 = self.subsampling_LSTM1(conv_input)
                dr_proj_lstm_output = self.Dropout_layer1(self.PROJ_Layer1(LSTM_output))
                dr_proj_lstm_output_ss = dr_proj_lstm_output[::2,:,:]
                #print(dr_proj_lstm_output_ss.shape)
                #-----------------------------------------
                LSTM_output2, hidden1 = self.subsampling_LSTM2(dr_proj_lstm_output_ss)
                dr_proj_lstm_output2 = self.Dropout_layer2(self.PROJ_Layer2(LSTM_output2))
                dr_proj_lstm_output_ss2 = dr_proj_lstm_output2[::2,:,:]
                #-----------------------------------------
                #print(dr_proj_lstm_output_ss2.shape)
                return dr_proj_lstm_output_ss2
##################################################################
##################################################################
###########################################################3    
class Res_LSTM_layers(nn.Module):
        def __init__(self, args):
                super(Res_LSTM_layers, self).__init__()

                self.hidden_size = args.hidden_size

                self.dropout = args.lstm_dropout
                self.isresidual = args.isresidual
                #------------------------------
                self.LSTM_layer = nn.LSTM(self.hidden_size,self.hidden_size,1,batch_first=False,bidirectional=True,dropout=self.dropout)
                self.PROJ_Layer = nn.Linear(self.hidden_size*2, self.hidden_size)
                self.Dropout_layer = nn.Dropout(p=self.dropout)
                #------------------------------
        def forward(self,lstm_ipt):
                #import pdb;pdb.set_trace()
                lstm_output, hidden1 = self.LSTM_layer(lstm_ipt)
                dr_proj_lstm_output = self.Dropout_layer(self.PROJ_Layer(lstm_output))
                
                ##residual connections
                if self.isresidual:
                    dr_proj_lstm_output = dr_proj_lstm_output + lstm_ipt
                return dr_proj_lstm_output
#=======================================================================
#=======================================================================
###########################################################3    
class Conv_Res_LSTM_Encoder(nn.Module):
        def __init__(self, args):
                super(Conv_Res_LSTM_Encoder, self).__init__()
                
                self.input_size = args.input_size
                self.hidden_size = args.hidden_size
                self.encoder_layers = args.encoder_layers
                
                self.lstm_dropout = args.lstm_dropout
                self.kernel_size = args.kernel_size
                
                self.stride = args.stride
                self.in_channels = args.in_channels
                self.out_channels = args.out_channels
                self.conv_dropout = args.conv_dropout
                self.isresidual = args.isresidual

                self.Input_subsamp_layers = subsampling_LSTMs(args)
                self.layer_stack = nn.ModuleList([Res_LSTM_layers(args) for _ in range(self.encoder_layers)])


        def forward(self, conv_res_ipt):
                #import pdb;pdb.set_trace()
                conv_res_ipt=self.Input_subsamp_layers(conv_res_ipt)
                for layer in self.layer_stack:
                        conv_res_ipt=layer(conv_res_ipt)
                        #print(conv_res_ipt.shape)
                return conv_res_ipt
#=======================================================================
#=======================================================================
#=============================================================================================
#=============================================================================================
# if __name__ == '__main__':




#     main()
# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)
# import pdb;pdb.set_trace()
# input=torch.rand(10,205,249)
# model=Conv_Res_LSTM_Encoder(args)
# Output=model(input)
# print(Output.shape)
