#!/usr/bin/pyhton
import sys
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument("--gpu",metavar='',type=int,default='0',help="use gpu flag 0|1")
#---------------------------

###model_parameter
parser.add_argument("--hidden_size",metavar='',type=int,default='320',help="Space token for the Char model")
parser.add_argument("--input_size",metavar='',type=int,default='249',help="Space token for the Char model")
parser.add_argument("--lstm_dropout",metavar='',type=float,default='0.3',help="lstm_dropout")
parser.add_argument("--kernel_size",metavar='',type=int,default='3',help="kernel_size")
parser.add_argument("--stride",metavar='',type=int,default='2',help="stride")
parser.add_argument("--in_channels",metavar='',type=int,default='1',help="in_channels")
parser.add_argument("--out_channels",metavar='',type=int,default='256',help="out_channels")
parser.add_argument("--conv_dropout",metavar='',type=float,default='0.1',help="conv_dropout")
parser.add_argument("--isresidual",metavar='',type=bool,default=True,help="isresidual --> 1|0 ")
parser.add_argument("--enc_front_end",metavar='',type=str,default='Subsamp_lstm',help="Subsamp_lstm|conv2d|nothing")
parser.add_argument("--Conv_Act",metavar='',type=str,default='relu',help="relu|tanh")

parser.add_argument("--nepochs",metavar='',type=int,default='100',help="No of epochs")
parser.add_argument("--encoder_dropout",metavar='',type=float,default='0.3',help="encoder dropout ")
parser.add_argument("--encoder_layers",metavar='',type=int,default='4',help="encoder dropout ")
parser.add_argument("--teacher_force",metavar='',type=float,default='0.6',help="Value of Teacher Force ")
parser.add_argument("--learning_rate",metavar='',type=float,default='0.0003',help="Value of learning_rate ")
parser.add_argument("--clip_grad_norm",metavar='',type=float,default='5',help="Value of clip_grad_norm ")
parser.add_argument("--new_bob_decay",metavar='',type=int,default='0',help="Value of new_bob_decay ")

#####Loss function parameters
parser.add_argument("--label_smoothing",metavar='',type=float,default='0.1',help="label_smoothing float value 0.1")
parser.add_argument("--use_speller",metavar='',type=bool,default=False,help="use_speller")
parser.add_argument("--use_word",metavar='',type=bool,default=True,help="use_word flags True|False")
parser.add_argument("--ctc_target_type",metavar='',type=str,default='word',help="ctc_target_type flags word|char")
parser.add_argument("--spell_loss_perbatch",metavar='',type=bool,default=False,help="ctc_target_type flags True|False")
parser.add_argument("--attention_type",metavar='',type=str,default='LAS',help="Attention type: LAS|Collin_monotonc|Location_aware")
parser.add_argument("--ctc_weight",metavar='',type=float,default=0.5,help="ctc weight")
parser.add_argument("--compute_ctc",metavar='',type=bool,default=True,help="compute ctc flags True|False")


####Training schedule parameters 
parser.add_argument("--no_of_checkpoints",metavar='',type=int,default='2',help="Flag of no_of_checkpoints ")
parser.add_argument("--tr_disp",metavar='',type=int,default='1000',help="Value of tr_disp ")
parser.add_argument("--vl_disp",metavar='',type=int,default='100',help="Value of vl_disp ")
parser.add_argument("--noise_inj_ratio",metavar='',type=float,default='0.1',help="Value of noise_inj_ratio ")
parser.add_argument("--weight_noise_flag",metavar='',type=str,default=True,help="T|F Flag for weight noise injection")

parser.add_argument("--early_stopping",metavar='',type=bool,default=False,help="Value of early_stopping ")
parser.add_argument("--early_stopping_checkpoints",metavar='',type=int,default=5,help="Value of early_stopping_checkpoints ")

parser.add_argument("--reduce_learning_rate_flag",metavar='',type=bool,default=True,help="reduce_learning_rate_flag True|False")
parser.add_argument("--lr_redut_st_th",metavar='',type=int,default=3,help="Value of lr_redut_st_th after this epochs the ls reduction gets applied")
#---------------------------


####bactching parameers
parser.add_argument("--model_dir",metavar='',type=str,default='models/Default_folder',help="model_dir")
parser.add_argument("--batch_size",metavar='',type=int,default='10',help="batch_size")
parser.add_argument("--max_batch_label_len",metavar='',type=int,default='50000',help="max_batch_label_len")
parser.add_argument("--max_batch_len",metavar='',type=int,default='20',help="max_batch_len")
parser.add_argument("--val_batch_size",metavar='',type=int,default='10',help="val_batch_size")

parser.add_argument("--validate_interval",metavar='',type=int,default='5000',help="steps")
parser.add_argument("--max_train_examples",metavar='',type=int,default='23380',help="steps")
parser.add_argument("--max_val_examples",metavar='',type=int,default='2039',help="steps")

parser.add_argument("--max_feat_len",metavar='',type=int,default='2000',help="max_seq_len the dataloader does not read the sequences longer that the max_feat_len, for memory and some times to remove very long sent for LSTM")
parser.add_argument("--max_label_len",metavar='',type=int,default='200',help="max_labes_len the dataloader does not read the sequences longer that the max_label_len, for memory and some times to remove very long sent for LSTM")

###plot the figures
parser.add_argument("--plot_fig_validation",metavar='',type=bool,default=False,help="True|False")
parser.add_argument("--plot_fig_training",metavar='',type=bool,default=False,help="True|False")

#**********************************
#Spec Aug
parser.add_argument("--spec_aug_flag",metavar='',type=bool,default=False,help="spec_aug_flag")
parser.add_argument("--min_F_bands",metavar='',type=int,default='30',help="min_F_bands")
parser.add_argument("--max_F_bands",metavar='',type=int,default='80',help="max_F_bands")
parser.add_argument("--time_drop_max",metavar='',type=int,default='4',help="time_drop_max")
parser.add_argument("--time_window_max",metavar='',type=int,default='4',help="time_window_max")
#**********************************

#---------------------------
####paths and tokenizers
parser.add_argument("--text_file",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/normalized_text_full_train_text',help="text transcription with dev and eval sentences")
parser.add_argument("--train_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/train/',help="model_dir")
parser.add_argument("--dev_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
parser.add_argument("--test_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/scp_files/dev/',help="model_dir")
#---------------------------
parser.add_argument("--Word_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__word.model',help="model_dir")
parser.add_argument("--Char_model_path",metavar='',type=str,default='/mnt/matylda3/vydana/benchmarking_datasets/Librispeech/fbankfeats/making_textiles/models_10K/Librispeech_960_TRAIN__char.model',help="model_dir")

####pretrained weights
#---------------------------
parser.add_argument("--pre_trained_weight",metavar='',type=str,default='0',help="pre_trained_weight if you dont have just give zero ")
parser.add_argument("--retrain_the_last_layer",metavar='',type=str,default='False',help="retrain_final_layer if you dont have just give zero ")
####load the weights
#---------------------------
parser.add_argument("--weight_text_file",metavar='',type=str,default='weight_folder/weight_file',help="weight_file")
parser.add_argument("--Res_text_file",metavar='',type=str,default='weight_folder/weight_file_res',help="Res_file")


####decoding_parameters
parser.add_argument("--RNNLM_model",metavar='',type=str,default='/mnt/matylda3/vydana/HOW2_EXP/Timit/models/TIMIT_fullnewsetup_2_4dr0.3_LAS_loc_arg_format_V2/model_architecture_',help="")
parser.add_argument("--LM_model",metavar='',type=str,default='None',help="LM_model")
parser.add_argument("--Am_weight",metavar='',type=float,default=1,help="lm_weight a float calue between 0 to 1 --->(Am_weight* Am_pred + (1-Am_weight)*lm_pred)")
parser.add_argument("--beam",metavar='',type=int,default=10,help="beam for decoding")
parser.add_argument("--gamma",metavar='',type=float,default=1,help="gamma (0-2), noisy eos rejection scaling factor while decoding")
parser.add_argument("--len_pen",metavar='',type=float,default=1,help="len_pen(0.5-2), len_pen maximum number of decoding steps")
parser.add_argument("--Decoding_job_no",metavar='',type=int,default=0,help="Res_file")
parser.add_argument("--scp_for_decoding",metavar='',type=int,default=0,help="scp file for decoding")
parser.add_argument("--plot_decoding_pics",metavar='',type=bool,default=True,help="T|F")
parser.add_argument("--decoder_plot_name",metavar='',type=str,default='default_folder',help="T|F")
#---------------------------
parser.add_argument("-v","--verbosity",action="count",help="increase output verbosity")





