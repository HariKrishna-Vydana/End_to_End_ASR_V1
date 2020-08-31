#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
from numpy.random import permutation
##------------------------------------------------------------------
import torch
from torch import autograd, nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
#----------------------------------------
from random import shuffle
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
os.environ['PYTHONUNBUFFERED'] = '1'
import glob
from statistics import mean
import json
import kaldi_io

#*************************************************************************************************************************
####### Loading the Parser and default arguments
#import pdb;pdb.set_trace()
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
import RNNLM_config
from RNNLM_config import parser
args = parser.parse_args()

###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)

#####setting the gpus in the gpu cluster
#**********************************
if args.gpu:
        cuda_command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv | tail -n+2 | ' \
               'awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
        oooo=subprocess.check_output(cuda_command, shell=True)

        dev_id=str(oooo).lstrip('b').strip("'").split('\\n')[0]
        os.environ["CUDA_VISIBLE_DEVICES"]=dev_id

        gpu_no=os.environ["CUDA_VISIBLE_DEVICES"]
        print("Using gpu number" + str(gpu_no))
        dummy_variable=torch.zeros((10,10))
dummy_variable = dummy_variable.cuda() if args.gpu else dummy_variable
#----------------------------------------------------------------
#=================================================================
#sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Dataloader_for_lm import DataLoader
from Load_sp_model import Load_sp_models
from Training_loop_rnnlm import train_val_model
from RNNLM import RNNLM
from utils__ import weights_init,reduce_learning_rate,read_as_list,gaussian_noise,plotting
from user_defined_losses import preprocess,compute_cer
#===================================================================
if not isdir(args.model_dir):
        os.makedirs(args.model_dir)

png_dir=args.model_dir+'_png'
if not isdir(png_dir):
        os.makedirs(png_dir)
############################################
#================================================================================
#=======================================================
def main():  
        ##
        ##INITILIZE THE RNNLM MODEL
        model = RNNLM(args)
        model = model.cuda() if args.gpu else model
        optimizer = optim.Adam(params=model.parameters(),lr=args.learning_rate,betas=(0.9, 0.99))
        print(model)


        ##Load setpiece models for Dataloaders
        Word_model=Load_sp_models(args.Word_model_path)
        Char_model=Load_sp_models(args.Char_model_path)
        ###initilize the model
        #============================================================
        #------------------------------------------------------------ 

        train_gen = DataLoader(files=glob.glob(args.train_path + "*"),
                                max_batch_label_len=20000,
                                max_batch_len=args.max_batch_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model)   
    

        dev_gen = DataLoader(files=glob.glob(args.dev_path + "*"),
                                max_batch_label_len=20000,
                                max_batch_len=args.max_batch_len,
                                max_label_len=args.max_label_len,
                                Word_model=Word_model)   

        val_history=np.zeros(args.nepochs)   
        #======================================
        for epoch in range(args.nepochs):
            ##start of the epoch
            tr_CER=[]; tr_BPE_CER=[]; L_train_cost=[]
            model.train();
            for trs_no in range(args.validate_interval):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"
                Output_trainval_dict=train_val_model(args = args, 
                                                    model = model,
                                                    optimizer = optimizer,
                                                    data_dict = B1,
                                                    trainflag = True)


                #get the losses form the dict
                SMP_CE=Output_trainval_dict.get('cost_cpu')
                L_train_cost.append(np.exp(SMP_CE))
                #==========================================
                if (trs_no%args.tr_disp==0):
                    print("tr ep:==:>",epoch,"sampl no:==:>",trs_no,"train_cost==:>",mean(L_train_cost),flush=True)    
                    #------------------------           
            ###validate the model
            #=======================================================
            model.eval()
            #=======================================================
            Vl_CER=[]; Vl_BPE_CER=[];L_val_cost=[]
            val_examples=0
            for vl_smp in range(args.max_val_examples):
                B1 = dev_gen.next()
                smp_feat = B1.get('smp_word_label')
                val_examples+=smp_feat.shape[0]
                assert B1 is not None, "None should never come out of the DataLoader"

                ##brak when the examples are more
                if (val_examples >= args.max_val_examples):
                    break;
                #--------------------------------------                
                Val_Output_trainval_dict=train_val_model(args=args,
                                                        model = model,
                                                        optimizer = optimizer,
                                                        data_dict = B1,
                                                        trainflag = False)
                

                SMP_CE=Val_Output_trainval_dict.get('cost_cpu')
                L_val_cost.append(np.exp(SMP_CE))
                #======================================================
                if (vl_smp%args.vl_disp==0) or (val_examples==args.max_val_examples-1):
                    print("val epoch:==:>",epoch,"val smp no:==:>",vl_smp,"val_cost:==:>",mean(L_val_cost),flush=True)                            
            #----------------------------------------------------
#==================================================================
            val_history[epoch]=(mean(L_val_cost))
            print("val_history:",val_history[:epoch+1])
            #================================================================== 
            ####saving_weights 
            ct="model_epoch_"+str(epoch)+"_sample_"+str(trs_no)+"_"+str(mean(L_train_cost))+"___"+str(mean(L_val_cost))
            print(ct)
            torch.save(model.state_dict(),join(args.model_dir,str(ct)))
            #######################################################                    

            ###open the file write and close it to avoid delays
            with open(args.weight_text_file,'a+') as weight_saving_file:
                print(join(args.model_dir,str(ct)), file=weight_saving_file)

            with open(args.Res_text_file,'a+') as Res_saving_file:
                print(float(mean(L_val_cost)), file=Res_saving_file)
            #=================================

            #early_stopping and checkpoint averaging: 
            if args.reduce_learning_rate_flag:
            #=================================================================
                A=val_history
                Non_zero_loss=A[A>0]
                min_cpts=np.argmin(Non_zero_loss)
                Non_zero_len=len(Non_zero_loss)

                if ((Non_zero_len-min_cpts) > 1) and epoch>args.lr_redut_st_th: #args.early_stopping_checkpoints:                                
                    reduce_learning_rate(optimizer)
                    #reduce_learning_rate(decoder_optim)

                    ###start regularization only when model starts to overfit
                    weight_noise_flag=True
                    spec_aug_flag=True
                #------------------------------------
                for param_group in optimizer.param_groups:
                    lr=param_group['lr']                
                    print("learning rate of the epoch:",epoch,"is",lr)   

                if args.early_stopping:
                    #------------------------------------
                    if lr<=1e-8:
                        print("lr reached to a minimum value")
                        exit(0)
            #----------------------------------
            #**************************************************************
            #**************************************************************
#=============================================================================================
#=============================================================================================
if __name__ == '__main__':
    main()



