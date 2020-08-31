#! /bin/sh
#
#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda3=0.5

#$ -o /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_fullnewsetup_argformat_2_4dr0.3_LAS_loc_WN_lrth20_TF0.6_check_v3.log
#$ -e /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_fullnewsetup_argformat_2_4dr0.3_LAS_loc_WN_lrth20_TF0.6_check_v3.log



PPATH="/mnt/matylda3/vydana/HOW2_EXP/Timit"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE


gpu=1
max_batch_len=10
tr_disp=50
vl_disp=10
validate_interval=500
max_val_examples=400

learning_rate=0.0001
early_stopping=False    
clip_grad_norm=5

hidden_size=320
n_layers=1
label_smoothing=0


#Attention type: LAS|Collin_monotonc|Location_aware
lr_redut_st_th=20


weight_noise_flag=False
reduce_learning_rate_flag=True
spec_aug_flag=False

pre_trained_weight="0"


plot_fig_validation=False
plot_fig_training=False
start_decoding=False
#---------------------------
Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'


text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
#text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_39phseq'

train_path='/mnt/matylda3/vydana/HOW2_EXP/Timit/LM_train_files/'
dev_path='/mnt/matylda3/vydana/HOW2_EXP/Timit/LM_train_files/'
test_path='/mnt/matylda3/vydana/HOW2_EXP/Timit/LM_train_files/'

model_file="TIMIT_fullnewsetup_2_4dr0.3_LAS_loc_arg_format_RNNLM"
model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res

mkdir -pv $model_dir

output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log

if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi
echo "$model_dir"
echo "$weight_file"
echo "$Res_file"
#---------------------------------------------------------------------------------------------
stdbuf -o0  python RNNLMLSTM_arg.py \
						--gpu $gpu \
						--text_file $text_file \
						--train_path $train_path \
						--dev_path $dev_path \
						--Word_model_path $Word_model_path \
						--Char_model_path $Char_model_path \
						--max_batch_len $max_batch_len \
						--tr_disp $tr_disp \
						--validate_interval $validate_interval \
						--n_layers $n_layers \
						--weight_text_file $weight_text_file \
						--Res_text_file $Res_text_file \
						--model_dir $model_dir \
						--max_val_examples $max_val_examples \
						--learning_rate $learning_rate \
						--early_stopping $early_stopping \
						--vl_disp $vl_disp \
						--clip_grad_norm $clip_grad_norm \
						--label_smoothing $label_smoothing \
						--lr_redut_st_th $lr_redut_st_th \
						--weight_noise_flag $weight_noise_flag \
						--pre_trained_weight $pre_trained_weight \
						--reduce_learning_rate_flag $reduce_learning_rate_flag \
						--lr_redut_st_th $lr_redut_st_th
#---------------------------------------------------------------------------------------------






