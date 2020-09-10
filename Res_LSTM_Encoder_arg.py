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

#=============================================================================================================
class Conv_2D_Layers(nn.Module):
        def __init__(self,args):
                super(Conv_2D_Layers,self).__init__()
                self.input_size = int(args.input_size)

                ##get the output as the same size of encoder d_model
                self.hidden_size = int(args.hidden_size)
                self.kernel_size = int(args.kernel_size)
                self.stride = args.stride
                self.in_channels = int(args.in_channels)
                self.out_channels = int(args.out_channels)
                self.conv_dropout  = args.conv_dropout              

                #self.input_layer_norm=torch.nn.LayerNorm(args.input_size)
                #dropout layer
                self.conv_dropout_layer = nn.Dropout(self.conv_dropout)

                ###two subsamling conv layers
                self.conv1= nn.Conv2d(in_channels=self.in_channels,
                                        out_channels=self.out_channels,
                                        kernel_size=self.kernel_size,
                                        stride=self.stride,
                                        padding=1, dilation=1, groups=1, bias=True)

                #self.conv1_norm = nn.LayerNorm(math.ceil(self.input_size/(self.stride)))
                
                self.conv2=torch.nn.Conv2d(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            stride=self.stride,
                                            padding=1, dilation=1, groups=1, bias=True)                

                linear_in_size=math.ceil(self.out_channels*(math.ceil(self.input_size/(self.stride*2))))

                #self.conv2_norm = nn.LayerNorm(math.ceil(self.input_size/(self.stride*2)))

                ### makes the outputs as  (B * T * d_model)
                self.linear_out=nn.Linear(linear_in_size, self.hidden_size)
                #self.linear_out_norm = nn.LayerNorm(self.hidden_size)

        def forward(self, input):
                conv_input=input.unsqueeze(1)
                CV1=F.relu(self.conv_dropout_layer(self.conv1(conv_input)))
                CV2=F.relu(self.conv_dropout_layer(self.conv2(CV1)))
                conv_output=CV2

                b, c, t, f = conv_output.size()
                conv_output=conv_output.transpose(1,2).contiguous().view(b,t,c*f)
                lin_conv_output=self.linear_out(conv_output)
                return lin_conv_output.transpose(0,1)
#---------------------------------------------------------------------------------------------------------------
#===============================================================================================================
###########################################################3    
class Res_LSTM_layers(nn.Module):
        def __init__(self, args):
                super(Res_LSTM_layers, self).__init__()
                self.hidden_size = args.hidden_size
                self.dropout = args.lstm_dropout
                self.isresidual = args.isresidual
                #------------------------------
                self.LSTM_layer = nn.LSTM(self.hidden_size,self.hidden_size,1,batch_first=False,bidirectional=True,dropout=self.dropout)
                #self.LSTM_Norm  = nn.LayerNorm(self.hidden_size*2)
                self.PROJ_Layer = nn.Linear(self.hidden_size*2, self.hidden_size)
                #self.PROJ_Layer_norm = nn.LayerNorm(self.hidden_size)

                self.Dropout_layer = nn.Dropout(p=self.dropout)
                #------------------------------
        def forward(self,lstm_ipt):
                #import pdb;pdb.set_trace()
                lstm_output, hidden1 = self.LSTM_layer(lstm_ipt)
                # lstm_output_norm = self.LSTM_Norm(lstm_output)
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
                self.enc_front_end = args.enc_front_end

                #---------------------------------------------------------
                if self.enc_front_end=='conv2d':
                        self.Input_subsamp_layers = Conv_2D_Layers(args)

                elif self.enc_front_end=='Subsamp_lstm':
                        self.Input_subsamp_layers = subsampling_LSTMs(args)
                else:
                    print("Choose a front end")
                    exit(0)
                #---------------------------------------------------------
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
