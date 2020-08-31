#!/usr/bin/python
import sys
import os
from os.path import join, isdir
#----------------------------------------
import glob
import json
from argparse import Namespace

#**********
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Initializing_model_LSTM_SS_v2_args import Initialize_Att_model
from Load_sp_model import Load_sp_models
from CMVN import CMVN
from utils__ import plotting
from user_defined_losses import compute_cer
from Decoding_loop import get_cer_for_beam
#-----------------------------------
import torch

import Attention_arg
from Attention_arg import parser
args = parser.parse_args()
model_path_name=join(args.model_dir,'model_architecture_')
print(model_path_name)
#--------------------------------
###load the architecture if you have to load
with open(model_path_name, 'r') as f:
        TEMP_args = json.load(f)

ns = Namespace(**TEMP_args)
args=parser.parse_args(namespace=ns)

if args.Am_weight < 1:
    ##model class
    from RNNLM import RNNLM

    ##config file for RNLM
    import RNNLM_config
    from RNNLM_config import parser

    # ###save architecture for decoding
    RNNLM_model_path_name=join(args.RNNLM_model,'model_architecture_')
    print("Using the language model in the path", RNNLM_model_path_name)
    with open(RNNLM_model_path_name, 'r') as f:
            RNNLM_TEMP_args = json.load(f)
    RNNLM_ns = Namespace(**RNNLM_TEMP_args)
    #RNNLM=parser.parse_args(namespace=RNNLM_ns)
    ##==================================
    RNNLM_ns.gpu=0
    LM_model=RNNLM(RNNLM_ns)
    LM_model.eval()
    LM_model = LM_model.cuda() if args.gpu else LM_model
    args.LM_model = LM_model
##==================================
##**********************************
##**********************************
def main():
        args.gpu=False
        best_weight_file=get_best_weights(args.weight_text_file,args.Res_text_file)
        print("best_weight_file",best_weight_file)
        args.pre_trained_weight=join(best_weight_file)

        #=================================================
        model,optimizer=Initialize_Att_model(args)
        model.eval()
        model = model.cuda() if args.gpu else model



        #=================================================
        plot_path=join(args.model_dir,'decoding_files','plots')
        if not isdir(plot_path):
                os.makedirs(plot_path)
        #=================================================
        ####read all the scps and make large scp with each lines as a feature
        decoding_files_list=glob.glob(args.dev_path + "*")
        scp_paths_decoding=[]
        for i_scp in decoding_files_list:
            scp_paths_decoding_temp=open(i_scp,'r').readlines()
            scp_paths_decoding+=scp_paths_decoding_temp

        #scp_paths_decoding this should contain all the scp files for decoding
        #====================================================
        ###sometime i tend to specify more jobs than maximum number of lines in that case python indexing error we get  
        job_no=int(args.Decoding_job_no)-1
        
        #args.gamma=0.5
        #print(job_no)
        #####get_cer_for_beam takes a list as input
        present_path=[scp_paths_decoding[job_no]]
        
        #print(present_path)
        text_file_dict = {line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(args.text_file)}
        get_cer_for_beam(present_path,model,text_file_dict,plot_path,args)

#--------------------------------
def get_best_weights(weight_text_file,Res_text_file):
        weight_saving_file = open(join(weight_text_file),'r')
        Res_saving_file = open(join(Res_text_file),'r')

        ##read weight_files and CER and pick the best weights
        weight_file_lines=weight_saving_file.readlines()
        res_file_lines=Res_saving_file.readlines()

        #import pdb;pdb.set_trace()
        res_values=[ float(i.strip()) for i in res_file_lines]
        best_value_index=res_values.index(min(res_values))
        best_weight_file=weight_file_lines[best_value_index]
        best_weight_file=best_weight_file.strip()
        return best_weight_file

#--------------------------------

if __name__ == '__main__':
    main()

