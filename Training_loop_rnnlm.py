#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from utils__ import weights_init,gaussian_noise
#---------------------------------------
def train_val_model(**kwargs):
        args = kwargs.get('args')
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
 
        trainflag = kwargs.get('trainflag')

        B1 = kwargs.get('data_dict')
        smp_word_label = B1.get('smp_word_label')
        smp_trans_text = B1.get('smp_trans_text')  

        #################finished expanding the keyword arguments#########
        ###################################################################
        optimizer.zero_grad() 
        Word_target=torch.LongTensor(smp_word_label)
        #--------------------------------
        Decoder_out_dict = model(Word_target)
        cost=Decoder_out_dict.get('cost')
        if trainflag:
                cost.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
                optimizer.step()
        #--------------------------------------
        cost_cpu = cost.item()
        #==================================================
        Output_trainval_dict={'cost_cpu':cost_cpu}
        return Output_trainval_dict
#=========================================================
