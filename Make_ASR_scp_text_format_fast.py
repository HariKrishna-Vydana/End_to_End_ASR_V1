#! /usr/bin/python

import sys
import os
from os.path import join


sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
from Load_sp_model import Load_sp_models
text_dlim=' @@@@ '
##================================================================
##================================================================
# output_file='Timit_text_like_MT'
# scp_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/sorted_feats_pdnn_train_scp'
# transcript='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# Translation='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_2'
# Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# Word_model = Load_sp_models(Word_model_path)
# Char_model = Word_model
# outfile=open(output_file,'w')
#=====================================================================================
def Search_for_utt(query, search_file,SPmodel):
        utt_text=search_file.get(query,'None')
        utt_text=" ".join(utt_text)

        if utt_text != 'None' and SPmodel:
                tokens_utt_text = SPmodel.EncodeAsIds(utt_text)
                tokens_utt_text = [str(intg) for intg in tokens_utt_text]
                tokens_utt_text = " ".join(tokens_utt_text)
        else:
                tokens_utt_text = 'None'
        
        utt_text = utt_text + text_dlim + tokens_utt_text + text_dlim
        return utt_text

#=======================================================================================
def format_tokenize_data(scp_file,transcript,Translation,outfile,Word_model,Char_model): 
        for scpfile in scp_file:      
          scp_dict={line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(scpfile)}
          transcript_dict={line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(transcript)}
          Translation_dict={line.split(' ')[0]:line.strip().split(' ')[1:] for line in open(Translation)}

          for query in list(scp_dict.keys()):
                inp_seq = query + text_dlim
                inp_seq += Search_for_utt(query, search_file=scp_dict,SPmodel=None)
                inp_seq += Search_for_utt(query, search_file=transcript_dict,SPmodel=Char_model)
                inp_seq += Search_for_utt(query, search_file=Translation_dict,SPmodel=Word_model)
                #------------------
                #print(inp_seq)
                print(inp_seq,file=outfile) 
#============================================================================
#format_tokenize_data([scp_file],transcript,Translation,outfile,Word_model,Char_model)
      
















