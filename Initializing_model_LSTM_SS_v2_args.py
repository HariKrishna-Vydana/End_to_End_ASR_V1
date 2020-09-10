#!/usr/bin/python
import sys
import os
from os.path import join, isdir
import torch
from torch import optim

from utils__ import weights_init,count_parameters
from Encoder_Decoder import Encoder_Decoder
#====================================================================================
def Initialize_Att_model(args):

        model = Encoder_Decoder(args)
        
        trainable_parameters=list(model.model_encoder.parameters()) + list(model.model_decoder.parameters())
        optimizer = optim.Adam(params=trainable_parameters,lr=args.learning_rate,betas=(0.9, 0.99))
        
        pre_trained_weight=args.pre_trained_weight
        weight_flag=pre_trained_weight.split('/')[-1]
        print("Initial Weights",weight_flag)
        
        if weight_flag != '0':
                print("Loading the model with the weights form:",pre_trained_weight)
                weight_file=pre_trained_weight.split('/')[-1]
                weight_path="/".join(pre_trained_weight.split('/')[:-1])

                enc_weight=join(weight_path,weight_file)
                model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=True)
        model= model.cuda() if args.gpu else model

        return model, optimizer
#====================================================================================
#====================================================================================
# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# model,optimizer=Initialize_Att_model(args)
# print(model)
# print(optimizer)
