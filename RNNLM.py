#!/usr/bin/python
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wtnrm

import numpy as np
# import keras
from keras.preprocessing.sequence import pad_sequences

import sys
import os
from os.path import join, isdir
sys.path.insert(0, '/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Load_sp_model import Load_sp_models

from CE_loss_label_smoothiong import cal_performance
from CE_loss_label_smoothiong import CrossEntropyLabelSmooth as cal_loss

from user_defined_losses import preprocess,compute_cer

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==========================================================
#==========================================================
class RNNLM(nn.Module):
        def __init__(self, args):
                super(RNNLM, self).__init__()

                self.Word_model = Load_sp_models(args.Word_model_path)

                ####word model
                self.targets_no = int(self.Word_model.__len__())
                self.pad_index  = self.targets_no
                self.sos_id     = self.targets_no + 1
                self.eos_id     = self.targets_no + 2
                self.mask_id    = self.targets_no + 3
                self.Wout_size  = self.targets_no + 4
                self.word_unk   = self.Word_model.unk_id()
                self.Word_SIL_tok   = self.Word_model.EncodeAsIds('_____')[0]

                #---------------------------------------
                self.use_gpu = args.gpu
                self.n_layers = args.n_layers
                self.hidden_size = args.hidden_size                
                self.emb_dim = args.hidden_size                
                self.label_smoothing = args.label_smoothing
                self.Rnnlm_dropout=0.1
                self.Rnnlm_embd_drop=nn.Dropout(0.1)
                self.normalize_length=False
                #---------------------------------------
                self.Wembedding     = nn.Embedding(self.Wout_size, self.hidden_size)
                self.Lang_Lstm    = nn.LSTM(self.hidden_size,self.hidden_size, self.n_layers ,batch_first=False,bidirectional=False,dropout=self.Rnnlm_dropout)#1
                self.W_Dist         = nn.Linear(self.hidden_size,self.Wout_size)
                #---------------------------------------
        def Initialize_hidden_states(self,batch_size):
                    h0=self.init_Hidden(batch_size)
                    h0=torch.cat([h0]*self.n_layers ,dim=0)
               
                    c0=self.init_Hidden(batch_size)
                    c0=torch.cat([c0]*self.n_layers ,dim=0)
                    return h0,c0


        def forward(self,Word_target,h0=None,c0=None):
                batch_size=Word_target.shape[0]

                if not torch.is_tensor(h0): 
                    h0, c0 = self.Initialize_hidden_states(batch_size)

                ###add sos and eos and padding
                Word_target_input,Word_target_output = preprocess(Word_target,self.pad_index,self.sos_id,self.eos_id)
                
                # Word_target   = Word_target.cuda() if self.use_gpu else Word_target
                Word_target_input = Word_target_input.cuda() if self.use_gpu else Word_target_input
                Word_target_output = Word_target_output.cuda() if self.use_gpu else Word_target_output
                #===================================================================
                
                ###empty lists to store the label predictions               
                cost=0
                embedded_vector=self.Rnnlm_embd_drop(self.Wembedding(Word_target_input))

                ####transpose for batch not first
                embedded_vector=embedded_vector.transpose(0,1)
                lang_output,(h0,c0)=self.Lang_Lstm(embedded_vector,(h0,c0))
                pred_out=self.W_Dist(lang_output)
                

                Word_target_output = Word_target_output.transpose(0,1).contiguous()

                ######CE label smoothing
                cost=cal_loss(pred_out.view(-1,pred_out.shape[2]), Word_target_output.view(-1),
                                IGNORE_ID=self.pad_index,normalize_length=self.normalize_length,smoothing=self.label_smoothing)
                numtokens=Word_target_output.view(-1).shape[0]
                cost=cost/numtokens
                #--------------------------------------------------------------------
                ###=====================================================================
                #
                
                Decoutput_dict={'cost':cost,
                                'output_seq':None,
                                'attention_record':None,
                                'Char_cer':100,
                                'Word_cer':100}
                return Decoutput_dict
        #--------------------------------------------       
        #--------------------------------------------
        def init_Hidden(self,batch_size):
                result = Variable(torch.zeros(1,batch_size,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def predict_rnnlm(self,word_prefix,h0,c0):
            embedded_vector=self.Rnnlm_embd_drop(self.Wembedding(word_prefix))
            embedded_vector=embedded_vector.transpose(0,1)
            lang_output,(h0_out,c0_out)=self.Lang_Lstm(embedded_vector,(h0,c0))
            pred_out=self.W_Dist(lang_output)
            return pred_out,(h0_out,c0_out)

            ####




#======================================================================================================================
#======================================================================================================================
# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/KAT_Attention')
# import RNNLM_config
# from RNNLM_config import parser
# args = parser.parse_args()
# print(args)

# Word_model=Load_sp_models(args.Word_model_path)

# Rnnlm=RNNLM(args)
# print(Rnnlm)


# Word_target = torch.randint(low=0, high=1000,size=(10,6))
# Word_target = torch.LongTensor(Word_target)
# Word_target = Variable(Word_target, requires_grad=False).contiguous()
# Rnnlm(Word_target)
# print(Word_target)





