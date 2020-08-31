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
        #exit(0)
        if weight_flag != '0':
                print("Loading the model with the weights form:",pre_trained_weight)
                weight_file=pre_trained_weight.split('/')[-1]
                weight_path="/".join(pre_trained_weight.split('/')[:-1])

                enc_weight=join(weight_path,weight_file)
                # dec_weight=join(weight_path,'decoder_'+weight_file)
                model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=True)
                # self.model_decoder.load_state_dict(torch.load(dec_weight, map_location=lambda storage, loc: storage),strict=True)
        model= model.cuda() if args.gpu else model



        # #-------------------------------------
        # model_encoder,model_decoder = Init_model_classes(args)              
        # ### Apply the weight initialization
        # #--------------------------------------
        # model_encoder.apply(weights_init)
        # model_decoder.apply(weights_init)
        # #====================================================================================
        # pre_trained_weight=args.pre_trained_weight
        # weight_flag=pre_trained_weight.split('/')[-1]
        # print(weight_flag)
        # #exit(0)
        # if weight_flag != '0':
        #         print("Loading the model with the weights form:",pre_trained_weight)
        #         weight_file=pre_trained_weight.split('/')[-1]
        #         weight_path="/".join(pre_trained_weight.split('/')[:-1])
                
        #         enc_weight=join(weight_path,weight_file)
        #         dec_weight=join(weight_path,'decoder_'+weight_file)

        #         model_encoder.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=False)
        #         model_decoder.load_state_dict(torch.load(dec_weight, map_location=lambda storage, loc: storage),strict=False)
        # #====================================================================================
        # # if retrain_final_layer=='True':
        # #         model.decoder.tgt_word_prj=torch.nn.Linear(model.decoder.d_model,out_features=n_tgt_vocab, bias=True)
        # #         model.decoder.tgt_word_emb=torch.nn.Embedding(n_tgt_vocab,model.decoder.d_model)

        # #weight='model_epoch_2_sample_2304_603.6812260614502___487.51988422993173___22.628098760522644'
        # ###this should be afte initializing or with switched off initializing
        # #--------------------------------------
        # #enc_weight=join(model_dir,weight)
        # #dec_weight=join(model_dir,'decoder_'+weight)
        # #model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage))
        # #model_decoder.load_state_dict(torch.load(dec_weight, map_location=lambda storage, loc: storage))
        # #--------------------------------------

        # model_encoder = model_encoder.cuda() if args.gpu else model_encoder
        # model_decoder = model_decoder.cuda() if args.gpu else model_decoder
        # #learning_rate=args.learning_rate       
        # encoder_optim=optim.Adam(params=model_encoder.parameters(),lr=args.learning_rate,betas=(0.9, 0.99))
        # decoder_optim=optim.Adam(params=model_decoder.parameters(),lr=args.learning_rate,betas=(0.9, 0.99))
        # #-------------------------------------- 
        # print("encoder:=====>",(count_parameters(model_encoder))/1000000.0)
        # print("decoder:=====>",(count_parameters(model_decoder))/1000000.0)

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
