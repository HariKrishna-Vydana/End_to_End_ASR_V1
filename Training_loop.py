#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Spec_Augument import Spec_Aug_freqCont as Spec_Aug
from utils__ import weights_init,gaussian_noise
#---------------------------------------
def train_val_model(**kwargs):

        args = kwargs.get('args')
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
 
        trainflag = kwargs.get('trainflag')
        weight_noise_flag = kwargs.get('weight_noise_flag')
        spec_aug_flag = kwargs.get('spec_aug_flag')

        B1 = kwargs.get('data_dict')
        smp_feat = B1.get('smp_feat')
        smp_char_label = B1.get('smp_char_label')
        smp_word_label = B1.get('smp_word_label')
        smp_trans_text = B1.get('smp_trans_text')  


        #################finished expanding the keyword arguments#########
        ##===========================================
        if args.spec_aug_flag and spec_aug_flag:
               smp_feat_mask = Spec_Aug(smp_feat,args.min_F_bands,args.max_F_bands,args.time_drop_max,args.time_window_max)
               smp_feat = smp_feat * smp_feat_mask

        # #==========================================
        if (args.weight_noise_flag) and weight_noise_flag:
                 params = list(model.parameters()) #+ list(model_decoder.parameters())
                 param = [gaussian_noise(param, args.gpu) for param in params]
        #============================================
        ###################################################################
        optimizer.zero_grad() 

        input=torch.from_numpy(smp_feat).float()

        Char_target=torch.LongTensor(smp_char_label)
        Word_target=torch.LongTensor(smp_word_label)

        #-----------------------------------------------------------------
        input=input.cuda() if args.gpu else input
        teacher_force_rate = args.teacher_force if trainflag else 0

        # H = model_encoder(input) 
        # ###encoder of the model
        # ###Decoder of the model         
        # Decoder_out_dict = model_decoder(H, teacher_force_rate, Char_target, Word_target, smp_trans_text)
        
        #--------------------------------
        Decoder_out_dict = model(input,teacher_force_rate,Char_target,Word_target,smp_trans_text)
        #--------------------------------
        cost=Decoder_out_dict.get('cost')
        if trainflag:
                cost.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)
                #torch.nn.utils.clip_grad_norm_(model_decoder.parameters(),args.clip_grad_norm)

                optimizer.step()
                #decoder_optim.step()
        #--------------------------------------
        cost_cpu = cost.item()
   
        ###output a dict
        attention_record = Decoder_out_dict.get('attention_record')[:,:,0].transpose(0,1)
        #==================================================
        Output_trainval_dict={
                            'cost_cpu':cost_cpu,
                            'attention_record':attention_record,
                            'Char_cer':Decoder_out_dict.get('Char_cer'),
                            'Word_cer':Decoder_out_dict.get('Word_cer')}
        return Output_trainval_dict
#=========================================================
