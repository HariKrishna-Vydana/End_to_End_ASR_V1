#!/usr/bin/python
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.weight_norm as wtnrm

import numpy as np
from keras.preprocessing.sequence import pad_sequences


from Load_sp_model import Load_sp_models
from CE_loss_label_smoothiong import cal_performance
from CE_loss_label_smoothiong import CrossEntropyLabelSmooth as cal_loss
from user_defined_losses import preprocess,compute_cer

import sys
import os
from os.path import join, isdir
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#==========================================================
#==========================================================
class decoder(nn.Module):
        def __init__(self, args):
                super(decoder, self).__init__()
                
                self.attention_type=args.attention_type

                # import pdb;pdb.set_trace()
                self.use_speller = args.use_speller
                self.use_word = args.use_word
                self.spell_loss_perbatch = args.spell_loss_perbatch
                self.ctc_target_type = args.ctc_target_type
                self.label_smoothing = args.label_smoothing

                if self.use_word:
                    print("print using the Words as a Targets")
                    self.Word_model = Load_sp_models(args.Word_model_path)
                    self.Char_model = Load_sp_models(args.Char_model_path)
                else :
                    print("print using the Char as a Targets")
                    self.Word_model = Load_sp_models(args.Char_model_path)
                    self.Char_model = Load_sp_models(args.Word_model_path)


                ####word model
                self.targets_no = int(self.Word_model.__len__())
                self.pad_index  = self.targets_no
                self.sos_id     = self.targets_no + 1
                self.eos_id     = self.targets_no + 2
                self.mask_id    = self.targets_no + 3
                self.Wout_size  = self.targets_no + 4
                self.word_unk   = self.Word_model.unk_id()
                self.Word_SIL_tok   = self.Word_model.EncodeAsIds('_____')[0]
                ####Char model
                #---------------------------------------
                self.Ch_tgts_no     = int(self.Char_model.__len__())
                self.Char_pad_id    = self.Ch_tgts_no
                self.Char_sos_id    = self.Ch_tgts_no + 1
                self.Char_eos_id    = self.Ch_tgts_no + 2
                self.Char_mask_id   = self.Ch_tgts_no + 3
                self.Char_out_size  = self.Ch_tgts_no + 4
                self.Ch_SIL_tok   = self.Char_model.EncodeAsIds('_____')[0]
                #---------------------------------------
                self.use_gpu = args.gpu
                self.hidden_size = args.hidden_size                
                self.emb_dim = args.hidden_size                

                #---------------------------------------
                #Word_model layers
                #---------------------------------------

                #ATTENT parameters
                kernel_size = 11 #kernal is always odd
                padding     = (kernel_size - 1) // 2

                self.conv   = nn.Conv1d(1, self.hidden_size, kernel_size, padding=padding)
                self.PSI    = nn.Linear(self.hidden_size,self.hidden_size)
                self.PHI    = nn.Linear(self.hidden_size,self.hidden_size)
                self.attn   = nn.Linear(self.hidden_size,1) 
                #---------------------------------------

                list_of_attentions_hdim=['LAS_LOC']
                self.Lexume_Lstm_input= self.hidden_size if (self.attention_type in list_of_attentions_hdim ) else self.hidden_size*2
                #-----------------------------------------
                ####### Lexume-Lstm
                self.Lexume_Lstm    = nn.LSTM(self.Lexume_Lstm_input,self.hidden_size, 1 ,batch_first=False,bidirectional=False)#1
                self.Wembedding     = nn.Embedding(self.Wout_size, self.hidden_size)
                self.W_Dist         = nn.Linear(self.hidden_size*2,self.Wout_size)
                #---------------------------------------          
                #---------------------------------------
                #---------------------------------------
                self.log_softmax = nn.Softmax(dim=1)      
                self.relu        = nn.ReLU()
                self.softmax     = nn.Softmax(dim=0)
                self.tanh        = nn.Tanh()
                #---------------------------------------
                self.ctc_weight = args.ctc_weight
                self.compute_ctc = args.compute_ctc

                if self.compute_ctc:
                    if self.ctc_target_type=='word':
                        self.CTC_output_layer = nn.Linear(self.hidden_size,self.Char_out_size)
                        self.CTC_Loss = torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
                    
                    elif self.ctc_target_type=='char':
                        self.CTC_output_layer = nn.Linear(self.hidden_size,self.Wout_size)
                        self.CTC_Loss = torch.nn.CTCLoss(blank=0,reduction='none',zero_infinity=True)
                    else:
                        print("ctc_target_type given wrong",self.ctc_target_type)
                        exit(0)
        #---------------------------------------
        #-------------------------------
        def select_step(self, H, yi, hn1, cn1, si, alpha_i_prev, ci):
                    if self.attention_type=='LAS':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_LAS(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    elif self.attention_type=='Collin_monotonc':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_Collin_Raffel_monotonic(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    elif self.attention_type=='Location_aware':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.step_Location_aware(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    elif self.attention_type=='LAS_LOC':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.LAS_LOC(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    elif self.attention_type=='LAS_LOC_ci':
                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.LAS_LOC_ci(H, yi, hn1, cn1, si, alpha_i_prev, ci)
                    #------------------------------
                    #-------------------------------
                    else:
                        print("-atention type undefined choose LAS |Collin_monotonc| Location_aware--->",self.attention_type)
                        exit(0)
                    return yout, alpha_i, si_out, hn1_out, cn1_out, ci_out
            #----------------------------------
        #===============================================
        def Additive_Att_LOC(self,H,si,alpha_i_prev):                        
                psi=self.PSI(si)
                phi=self.PHI(H)               

                pfi=self.conv(alpha_i_prev.transpose(0,2)).transpose(0,2).transpose(1,2)
                
                ei=self.attn(self.tanh(phi + psi.expand_as(phi) + pfi))        
                
                alpha_i=self.softmax(ei.squeeze(2)).unsqueeze(1)
                
                ci=torch.bmm(alpha_i.transpose(2,0),H.transpose(0,1))
                ci=ci.transpose(0,1)

                ci_pl_si=torch.cat([ci,si],2)
                return ci,alpha_i,ci_pl_si,phi

        #===============================================
        def Additive_Att(self,H,si,alpha_i_prev):                        
                psi=self.PSI(si)
                phi=self.PHI(H)               
               
                ei=self.attn(self.tanh(phi + psi.expand_as(phi)))                        
                alpha_i=self.softmax(ei.squeeze(2)).unsqueeze(1)

                ci=torch.bmm(alpha_i.transpose(2,0),H.transpose(0,1))
                ci=ci.transpose(0,1)

                ci_pl_si=torch.cat([ci,si],2)
                return ci,alpha_i,ci_pl_si,phi
        #----------------------------------------------------------------------------------
        #-----------------------------------------------------------------------------------
        #------------------------------------------------------------------------------------
        def step_Collin_Raffel_monotonic(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####LAS model has no LOC_ATT
                #  ci_out=Additive_Att(si,H); si_out=RNN(si, [yi,ci_out]);  yout=W_Dist([si_out,ci_out])
                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att(H,si,alpha_i_prev)

                lstm_input=torch.cat([yi,ci_out],2)
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                
                yout=self.W_Dist(torch.cat([ci_out,si_out],dim=2))
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out

        def step_LAS(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####LAS model has no LOC_ATT
                #  si_out=RNN(si, [yi,ci]); ci_out=Additive_Att(si_out,H); yout=W_Dist([si_out,ci_out])
                lstm_input=torch.cat([yi,ci],2)
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att(H,si_out,alpha_i_prev)
                yout=self.W_Dist(ci_pl_si)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out

        #----------------------------------------------------------------------------------------
        def step_Location_aware(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                ####Ci is not used here
                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si,alpha_i_prev)
                
                lstm_input=torch.cat([yi,ci_out],2)
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))
                
                #### op is from si and ci_out  
                lstm_output=torch.cat([si,ci_out],2) ####this is different from others its not [si_out,ci_out]
                yout=self.W_Dist(lstm_output)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        def LAS_LOC(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                #stm_input=torch.cat([yi,yi],dim=2)
                lstm_input=yi
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si_out,alpha_i_prev)
                yout=self.W_Dist(ci_pl_si)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        def LAS_LOC_ci(self,H,yi,hn1,cn1,si,alpha_i_prev,ci):
                lstm_input=torch.cat([yi,ci],dim=2)
                #lstm_input=yi
                si_out,(hn1_out,cn1_out)=self.Lexume_Lstm(lstm_input,(hn1,cn1))

                ci_out,alpha_i,ci_pl_si,phi=self.Additive_Att_LOC(H,si_out,alpha_i_prev)
                yout=self.W_Dist(ci_pl_si)
                return yout,alpha_i,si_out,hn1_out,cn1_out,ci_out
        #----------------------------------------------------------------------------------
        #=======================================================================================
        def forward(self,H,teacher_force_rate,Char_target,Word_target,L_text):
                #----------
                #if not self.use_word:
                #     Char_target, Word_target = Word_target, Char_target
                ###add sos and eos and padding
                _,Char_target = preprocess(Char_target,self.Char_pad_id,self.Char_sos_id,self.Char_eos_id)
                _,Word_target = preprocess(Word_target,self.pad_index,self.sos_id,self.eos_id)
                
                Char_target   = Char_target.cuda() if self.use_gpu else Char_target
                Word_target   = Word_target.cuda() if self.use_gpu else Word_target
                #===================================================================
                ###empty lists to store the label predictions
                output_seq , attention_record, greedy_label_seq  = [], [], []
                decoder_steps           = Word_target.size(1)
                batch_size              = H.size(1)
                AUCO_SEQ_len            = H.size(0)
                cost=0
                #--------------------------------------------------------------------
                ####initiaizing the decoder LSTMs and attentions 
                pred_label, yi, si, hn1, cn1, ci, alpha_i_prev =self.initialize_decoder_states(batch_size,AUCO_SEQ_len)
                #---------------------------------------------------------------------------
                #---------------------------------------------------------------------------
                for d_steps in range(decoder_steps):

                        yout, alpha_i, si_out, hn1_out, cn1_out, ci_out = self.select_step(H, yi, hn1, cn1, si, alpha_i_prev,ci) 
                        pred_out=F.softmax(yout,2)                                  
                        #--------------------------------------------
                        #-------------------------------------------- 
                        #cost+=loss(yout.squeeze(0),Word_target[:,d_steps]) + char_lstm_loss
                        present_word_cost = cal_loss(yout.squeeze(0),Word_target[:,d_steps],self.pad_index, normalize_length=True, smoothing=self.label_smoothing)
                        cost += present_word_cost

                        #-----------------------------------------

                        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
                        if teacher_force:
                                pred_label=Word_target[:,d_steps]
                        else:
                                pred_label=torch.argmax(pred_out,2)
                                pred_label=pred_label.squeeze(0)
                        #-----------------------------------------
                        yi=self.Wembedding(pred_label).unsqueeze(0)

                        hn1, cn1, si, ci, alpha_i_prev = hn1_out, cn1_out, si_out, ci_out, alpha_i 

                        output_seq.append(pred_out)
                        attention_record.append(alpha_i)
                ###=====================================================================
                ###Normalize the loss per_label and not per batch

                #print("Att_cost before normalized",cost)
                cost = cost/(decoder_steps*batch_size)
                #print("Att_cost after normalized",cost)
                ###=====================================================================
                #Computing WER
                #import pdb; pdb.set_trace()
                attention_record = torch.cat(attention_record,dim=1)
                output_seq = torch.cat(output_seq,dim=0).transpose(1,0)
                output_seq = torch.argmax(output_seq,2)
                
                #####
                ####The transformer type error measure does not correspond between training and testing
                #other two error does not corrspond with the error of the systems
                OP=output_seq.data.cpu().numpy(); OP1=np.asarray(OP.flatten())
                WP=Word_target.data.cpu().numpy(); WP1=np.asarray(WP.flatten())
                Word_cer=compute_cer(WP1,OP1,self.pad_index)

                ###======================================================================
                #import pdb; pdb.set_trace()
                Char_CTC_loss=0
                if self.compute_ctc:
                    #breakpoint()
                    ctc_output = self.CTC_output_layer(H)
                    #ctc_output = ctc_output.transpose(0,1)
                    log_probs=torch.nn.functional.log_softmax(ctc_output,dim=2)
                    input_lengths  = torch.IntTensor([H.size(0)],).repeat(H.size(1))
                    target_lengths = torch.IntTensor([Char_target.size(1),]).repeat(Char_target.size(0))
                    input_lengths  =   Variable(input_lengths, requires_grad=False).contiguous()
                    target_lengths =  Variable(target_lengths, requires_grad=False).contiguous()
                    input_lengths   = input_lengths.cuda() if self.use_gpu else input_lengths
                    target_lengths  = target_lengths.cuda() if self.use_gpu else target_lengths
                    Char_CTC_loss   = self.CTC_Loss(log_probs,Char_target,input_lengths,target_lengths)
                    CTC_norm_factor = Char_target.size(1)*batch_size

                    #print("CTC_cost before the norm factor",Char_CTC_loss,Char_CTC_loss.sum())
                    #### check the normalizations ##mean acorss batch and normalize per-frame
                    Char_CTC_loss = Char_CTC_loss.sum()/CTC_norm_factor
                    #print("CTC_cost after the norm factor",Char_CTC_loss)
                    ###======================================================================
                    cost = Char_CTC_loss * self.ctc_weight + cost * (1-self.ctc_weight)
                ###=====================================================================
                ###=====================================================================
                #
                #

                Char_cer = Word_cer
                Decoutput_dict={'cost':cost,
                                'output_seq':output_seq,
                                'attention_record':attention_record,
                                'Char_cer':Char_cer,
                                'Word_cer':Word_cer}
                return Decoutput_dict
        #---------------------------------------------------------------------------------------------------------
        #---------------------------------------------------------------------------------------------------------
#======================================================================================================================
        def decode_with_beam_LM(self,H,LM_model,Am_weight,beam,gamma,len_pen):
                # breakpoint()
                batch_size = H.size(1)
                AUCO_SEQ_len = H.size(0)
                output_seq , attention_record, greedy_label_seq = [], [], []
                #---------------------------------------------------------------------------------------
                if batch_size !=1:
                        print("does not support batch_size greater than 1")
                        exit()
                #---------------------------------------------------------------------------------------
                max_len=AUCO_SEQ_len*len_pen
                
                pred_label, yi, si, hn1, cn1, ci, alpha_i_prev = self.initialize_decoder_states(batch_size,AUCO_SEQ_len)
                #---------------------------------------------------------------------------
                
                if Am_weight < 1:
                    rnnlm_h0,rnnlm_c0=LM_model.Initialize_hidden_states(batch_size)
                else:
                     rnnlm_h0, rnnlm_c0 = 0, 0
                     rnnlm_h0_out, rnnlm_c0_out = 0, 0

                
                #---------------------------------------------------------------------------                
                ys = torch.ones(1,batch_size).fill_(self.sos_id).type_as(H).long()


                hyp = {'score': 0.0, 'yseq': ys,'state': [hn1,cn1,si,alpha_i_prev,ci,rnnlm_h0,rnnlm_c0],'alpha_i_list':alpha_i_prev}
                hyps = [hyp]
                ended_hyps = []

                for d_steps in range(max_len):
                        hyps_best_kept=[]
                        for hyp in hyps:
                                ys=hyp['yseq']
                                # if Am_weight<1:
                                #     lm_predict_out=LM_model(ys)[:,-1,:].unsqueeze(1)
                                #--------------------------- 

                                pred_label = pred_label if d_steps==0 else ys[:,-1].unsqueeze(1)
                                yi=self.Wembedding(pred_label)
                                #-------------------------------
                                ####no_teacher forcing so always predicted label

                                present_state=hyp['state']
                                [hn1,cn1,si,alpha_i_prev,ci,rnnlm_h0,rnnlm_c0]=present_state
                                
                                yout,alpha_i,si_out,hn1_out,cn1_out,ci_out=self.select_step(H,yi,hn1,cn1,si,alpha_i_prev,ci)



                                # new_state= [hn1_out,cn1_out,si_out,alpha_i,ci_out,rnnlm_h0_out,rnnlm_c0_out]
                                # hyp['state']=new_state

                                Am_pred = F.log_softmax(yout,2)
                                #Am_pred = Am_pred #-1.0 #####insertion_penalty 

                                if Am_weight < 1:    
                                    #breakpoint()
                                    ys_lm_input = ys[:,-1].unsqueeze(0) if d_steps>0 else ys
                                    lm_predict_out,(rnnlm_h0_out, rnnlm_c0_out)=LM_model.predict_rnnlm(ys_lm_input,h0=rnnlm_h0, c0=rnnlm_c0)
                                    Lm_pred = F.log_softmax(lm_predict_out,2)
                                    pred_out = Am_weight*Am_pred + (1-Am_weight)*Lm_pred
                                else:
                                    pred_out=Am_pred



                                ####set new states after am and LM predictions
                                new_state= [hn1_out,cn1_out,si_out,alpha_i,ci_out,rnnlm_h0_out,rnnlm_c0_out]
                                hyp['state']=new_state



                                #--------------------------------------
                                #beam-------code
                                local_best_scores, local_best_ids = torch.topk(pred_out, beam, dim=2)
                                #print(local_best_scores.size(),local_best_ids.size())
                                #Eos threshold
                                # pdb.set_trace()
                                ####------------------------------------------------------------------
                                #breakpoint()
                                #print(ys)
                                EOS_mask=local_best_ids==self.eos_id
                                if (EOS_mask.any()) and beam>1:
                                    KEEP_EOS=local_best_scores[EOS_mask] > gamma * torch.max(local_best_scores[~EOS_mask])
                                    #print(KEEP_EOS)
                                    if (KEEP_EOS.item()):
                                        pass;
                                    else:
                                        local_best_scores[EOS_mask]=-1000
                                #print(local_best_scores,local_best_ids)
                                ####------------------------------------------------------------------                             
                                ####------------------------------------------------------------------
                                all_candidates=[]
                                for j in range(beam):
                                        new_hyp = {}

                                        present_lab=torch.tensor([[int(local_best_ids[0,0,j])]])
                                        new_hyp['score'] = hyp['score'] + local_best_scores[0,0,j]
                                        present_lab=present_lab.cuda() if H.is_cuda else present_lab 
                                        new_hyp['yseq'] = torch.cat((ys,present_lab),dim=1)
                                        new_hyp['state'] = new_state
                                        new_hyp['alpha_i_list'] = torch.cat((hyp['alpha_i_list'],new_state[3]),dim=1)

                                        hyps_best_kept.append(new_hyp)
                                        #print(new_hyp['yseq'],new_hyp['score'])
                                #========================================
                                hyps_best_kept = sorted(hyps_best_kept,key=lambda x: x['score'],reverse=True)[:beam]
                                #print(hyps_best_kept['yseq'],hyps_best_kept['score'])
                                #===============================================
                                #print(hyps_best_kept)

                        #===============================================
                        remained_hyps = []
                        for hyp in hyps_best_kept:
                                hyp_len=hyp['yseq'][0].size()[0]
                                if hyp['yseq'][0, -1].item() == self.eos_id and d_steps>0:
                                        ended_hyps.append(hyp)
                                else:
                                        remained_hyps.append(hyp)

                        hyps = remained_hyps


                ### add the unfinishe hyps to finished hypes at the end of max_len
                if len(ended_hyps)==0:
                        ended_hyps=remained_hyps

                ### sort the the hypds bases on score
                nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), beam)]
                #-----------------------
                

                ### Convert the hyps to text and add the text seq to dict
                output_dict=[]
                for OP in nbest_hyps:
                    OP.pop('state')
                    OP['Text_seq']=self.get_charecters_for_sequences(OP['yseq'])
                    output_dict.append(OP)
                #-------------------------------------
                #-------------------------------------                
                #-------------------------------------                                                 
                return output_dict
                #=======================================================================================

        #--------------------------------------------
        def get_charecters_for_sequences(self,input_tensor):
            """ Takes pytorch tensors as in put and print the text charecters as ouput,  
            replaces sos and eos as unknown symbols and ?? and later deletes them from the output string"""
            output_text_seq=[]
            final_token_seq=input_tensor.data.numpy()
            final_token_seq=np.where(final_token_seq>=self.pad_index,self.Word_SIL_tok,final_token_seq)
            text_sym_sil_tok=self.Word_model.DecodeIds([self.Word_SIL_tok])
            
            for i in final_token_seq:
                i=i.astype(np.int).tolist()
                text_as_string=self.Word_model.DecodeIds(i)
                text_as_string=text_as_string.replace(text_sym_sil_tok,"")
                output_text_seq.append(text_as_string)
            return output_text_seq

        #=======================================================================================
        def initialize_decoder_states(self,batch_size,AUCO_SEQ_len):
                pred_label = torch.ones(1,batch_size).fill_(self.sos_id).long()
                pred_label = pred_label.cuda() if self.use_gpu else pred_label

                yi  = self.Wembedding(pred_label)
                si  = self.init_Hidden(batch_size) ######si=torch.mean(H,0,keepdim=True) could also be used
                hn1 = self.init_Hidden(batch_size)
                cn1 = self.init_Hidden(batch_size)
                ci  = self.init_Hidden(batch_size)
                alpha_i_prev = self.init_LOC_Att_vec(batch_size,AUCO_SEQ_len)
                return pred_label, yi, si, hn1, cn1, ci, alpha_i_prev
        #--------------------------------------------
        #--------------------------------------------
        def init_Hidden(self,batch_size):
                result = Variable(torch.zeros(1,batch_size,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_Output(self,batch_size):
                result = Variable(torch.zeros(1,batch_size,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_Att_vec(self):
                result = Variable(torch.zeros(1,1,self.hidden_size))
                result=result.cuda() if self.use_gpu else result
                return result
       #--------------------------------------------
        def init_LOC_Att_vec(self,B,n):
                result = Variable(torch.zeros(n,1,B))
                result=result.cuda() if self.use_gpu else result
                return result
        #--------------------------------------------
        def init_embeding_vector(self):
                result = Variable(torch.zeros(1,1,self.emb_dim))
                result=result.cuda() if self.use_gpu else result
                return result
        #-------------------------------------------
#======================================================================================================================

#======================================================================================================================
#======================================================================================================================
# ##debugger
# sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
# import Attention_arg
# from Attention_arg import parser
# args = parser.parse_args()
# print(args)


# #import pdb;pdb.set_trace()
# # Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__word.model'
# # Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__char.model'
# # text_file = '/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/normalized_text_full_train_text_100lines'

# args.Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# args.text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# text_file = args.text_file

# H = torch.rand((81,10,320))      #torch.randint(low=0, high=1000,size=(10,6))
# H = torch.Tensor(H)
# H = Variable(H, requires_grad=False)

# #torch.rand((81,10,320))
# #import pdb;pdb.set_trace()

# # Word_target=torch.randint(low=0, high=1000,size=(10,6))
# # Char_target=torch.randint(low=0, high=20,size=(10,25))

# Word_target = torch.randint(low=0, high=62,size=(10,6))
# Word_target = torch.LongTensor(Word_target)
# Word_target = Variable(Word_target, requires_grad=False).contiguous().long()


# Char_target = torch.randint(low=0, high=20,size=(10,25))
# Char_target = torch.LongTensor(Char_target)
# Char_target = Variable(Char_target, requires_grad=False).contiguous().long()


# text_dict = {line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(text_file)}
# text_trans_list = [text_dict.get(T) for T in text_dict.keys()]
# text_trans_list_length = [len(text_dict.get(T)) for T in text_dict.keys()]
# L_text = pad_sequences(text_trans_list,maxlen=max(text_trans_list_length),dtype=object,padding='post',value='unk')
# #L_text = L_text[:10] 



# # import sentencepiece as spm
# # Word_model = spm.SentencePieceProcessor()
# # Char_model = spm.SentencePieceProcessor()
# # Word_model.Load(join(Word_model_path))
# # Char_model.Load(join(Char_model_path))
# #import pdb;pdb.set_trace()

# hidden_size = 320;
# compute_ctc = True
# ctc_weight  = 0.5
# use_gpu = 0

# use_speller = False
# use_word = True
# ctc_target_type='word'
# teacher_force_rate=0.6
# spell_loss_perbatch=False
# label_smoothing=0.1
# #import pdb;pdb.set_trace()

# args.attention_type='LAS'
# Dec=decoder(args)
# print(Dec)

# #decoder_weight="/mnt/matylda3/vydana/HOW2_EXP/Timit/models/Timit_Conv_Res_LSTM_3layers_256_LSTMSS_ls0.1/decoder_model_epoch_49_sample_17151_7.7726516959396275___1205.532112121582__0.4075471698113208"
# #Dec.load_state_dict(torch.load(decoder_weight, map_location=lambda storage, loc: storage),strict=True)

# H = H.cuda() if use_gpu else H

# Dec(H,teacher_force_rate,Char_target,Word_target,L_text)
# LM_model=None
# Am_weight=1
# beam=5
# gamma=1
# len_pen=1
# #print(H[:,0,:].unsqueeze(1).shape)
# #exit(0)
# #
# #BEAM_OUTPUT=Dec.decode_with_beam_LM(H[:,0,:].unsqueeze(1),LM_model,Am_weight,beam,gamma,len_pen)
# #print(BEAM_OUTPUT)
# print('-------Over---------')



