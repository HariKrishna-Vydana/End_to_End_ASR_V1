#!/usr/bin/pyhton
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--gpu",metavar='',type=int,default='0',help="use gpu flag 0|1")
#---------------------------
###model_parameter
parser.add_argument("--hidden_size",metavar='',type=int,default='320',help="Space token for the Char model")
parser.add_argument("--n_layers",metavar='',type=int,default='5',help="n_layers")
parser.add_argument("--lstm_dropout",metavar='',type=float,default='0.3',help="lstm_dropout")
parser.add_argument("--isresidual",metavar='',type=bool,default=True,help="isresidual --> 1|0 ")
#---------------------------
parser.add_argument("--nepochs",metavar='',type=int,default='100',help="No of epochs")
parser.add_argument("--learning_rate",metavar='',type=float,default='0.0003',help="Value of learning_rate ")
parser.add_argument("--clip_grad_norm",metavar='',type=float,default='5',help="Value of clip_grad_norm ")
parser.add_argument("--new_bob_decay",metavar='',type=int,default='0',help="Value of new_bob_decay ")
parser.add_argument("--no_of_checkpoints",metavar='',type=int,default='2',help="Flag of no_of_checkpoints ")
parser.add_argument("--weight_noise_flag",metavar='',type=str,default=True,help="T|F Flag for weight noise injection")

parser.add_argument("--early_stopping",metavar='',type=bool,default=False,help="Value of early_stopping ")
parser.add_argument("--early_stopping_checkpoints",metavar='',type=int,default=5,help="Value of early_stopping_checkpoints ")

parser.add_argument("--reduce_learning_rate_flag",metavar='',type=bool,default=True,help="reduce_learning_rate_flag True|False")
parser.add_argument("--lr_redut_st_th",metavar='',type=int,default=3,help="Value of lr_redut_st_th after this epochs the ls reduction gets applied")
#---------------------------
####bactching parameers
parser.add_argument("--Rnnlm_model_dir",metavar='',type=str,default='models/Default_folder',help="model_dir")
parser.add_argument("--batch_size",metavar='',type=int,default='10',help="batch_size")
parser.add_argument("--max_batch_label_len",metavar='',type=int,default='50000',help="max_batch_label_len")
parser.add_argument("--max_batch_len",metavar='',type=int,default='20',help="max_batch_len")
parser.add_argument("--val_batch_size",metavar='',type=int,default='10',help="val_batch_size")
parser.add_argument("--max_label_len",metavar='',type=int,default='200',help="max_label_len")


parser.add_argument("--tr_disp",metavar='',type=int,default='1000',help="Value of tr_disp ")
parser.add_argument("--vl_disp",metavar='',type=int,default='100',help="Value of vl_disp ")

parser.add_argument("--label_smoothing",metavar='',type=float,default='0.1',help="label_smoothing float value 0.1")
parser.add_argument("--validate_interval",metavar='',type=int,default='5000',help="steps")
parser.add_argument("--max_train_examples",metavar='',type=int,default='23380',help="steps")
parser.add_argument("--max_val_examples",metavar='',type=int,default='2039',help="steps")
#---------------------------


####paths and tokenizers
parser.add_argument("--text_file",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/normalized_text_full_train_text',help="text transcription with dev and eval sentences")
parser.add_argument("--train_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/train/',help="model_dir")
parser.add_argument("--dev_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
parser.add_argument("--test_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
#---------------------------


parser.add_argument("--Word_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__word.model',help="model_dir")
parser.add_argument("--Char_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__char.model',help="model_dir")


parser.add_argument("--Word_model",metavar='',type=str,default='Nothing',help="model_dir")
parser.add_argument("--Char_model",metavar='',type=str,default='Nothing',help="model_dir")
#---------------------------
parser.add_argument("--RNNLM_pre_trained_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")
parser.add_argument("--retrain_the_last_layer",metavar='',type=str,default='False',help="retrain_final_layer if you dont have just give zero ")
#---------------------------
parser.add_argument("--weight_text_file",metavar='',type=str,default='weight_folder/weight_file',help="weight_file")
parser.add_argument("--Res_text_file",metavar='',type=str,default='weight_folder/weight_file_res',help="Res_file")

#---------------------------
parser.add_argument("-v","--verbosity",action="count",help="increase output verbosity")





