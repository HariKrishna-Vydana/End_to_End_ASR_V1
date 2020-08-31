#! /bin/sh
#
#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda3=0.5

#$ -o /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_fullnewsetupV2_argformat_2_4dr0.3_LAS_loc_WN_lrth20_TF0.6.log
#$ -e /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_fullnewsetupV2_argformat_2_4dr0.3_LAS_loc_WN_lrth20_TF0.6.log




PPATH="/mnt/matylda3/vydana/HOW2_EXP/Timit"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE



gpu=1
max_batch_len=10
tr_disp=50
vl_disp=10
validate_interval=500
max_val_examples=400

compute_ctc=False
ctc_weight=0.3

learning_rate=0.0001
early_stopping=False    
clip_grad_norm=5

hidden_size=320
input_size=249
encoder_layers=4
lstm_dropout=0.3
kernel_size=3
stride=2
in_channels=1
out_channels=64
conv_dropout=0.3
isresidual=False
label_smoothing=0.1


#Attention type: LAS|Collin_monotonc|Location_aware
attention_type='LAS_LOC'
lr_redut_st_th=20
teacher_force=0.6

min_F_bands=5
max_F_bands=30
time_drop_max=2
time_window_max=1

weight_noise_flag=False
reduce_learning_rate_flag=True
spec_aug_flag=False

#pre_trained_weight="/mnt/matylda3/vydana/HOW2_EXP/Timit/models/TIMIT_fullnewsetup_2_4dr0.3_LAS_loc/model_epoch_13_sample_6501_40.030520648190546___1500.3319549560547__0.23846153846153847"
pre_trained_weight="0"


plot_fig_validation=True
plot_fig_training=True
start_decoding=True


#---------------------------
Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'


#text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_39phseq'
train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'

dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'
test_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/test/'

model_file="TIMIT_fullnewsetup_2_4dr0.3_LAS_loc_arg_format_V2"
model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res
mkdir -pv $model_dir
#weight="$model_dir/$weight"


output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log

if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi
echo "$model_dir"
echo "$weight_file"
echo "$Res_file"


#---------------------------------------------------------------------------------------------
stdbuf -o0  python TIMIT_Att_V1_LSTMSS_v2_arg.py \
						--gpu $gpu \
						--text_file $text_file \
						--train_path $train_path \
						--dev_path $dev_path \
						--Word_model_path $Word_model_path \
						--Char_model_path $Char_model_path \
						--max_batch_len $max_batch_len \
						--tr_disp $tr_disp \
						--validate_interval $validate_interval \
						--encoder_layers $encoder_layers \
						--weight_text_file $weight_text_file \
						--Res_text_file $Res_text_file \
						--model_dir $model_dir \
						--max_val_examples $max_val_examples \
						--compute_ctc $compute_ctc \
						--ctc_weight $ctc_weight \
						--spec_aug_flag $spec_aug_flag \
						--in_channels $in_channels \
						--out_channels $out_channels \
						--learning_rate $learning_rate \
						--early_stopping $early_stopping \
						--vl_disp $vl_disp \
						--clip_grad_norm $clip_grad_norm \
						--label_smoothing $label_smoothing \
						--attention_type $attention_type \
						--lr_redut_st_th $lr_redut_st_th \
						--min_F_bands $min_F_bands \
						--max_F_bands $max_F_bands \
						--time_drop_max $time_drop_max \
						--time_window_max $time_window_max \
						--weight_noise_flag $weight_noise_flag \
						--teacher_force $teacher_force \
						--pre_trained_weight $pre_trained_weight \
						--reduce_learning_rate_flag $reduce_learning_rate_flag \
						--lr_redut_st_th $lr_redut_st_th \
						--plot_fig_validation $plot_fig_validation \
						--plot_fig_training $plot_fig_training \
#---------------------------------------------------------------------------------------------


# gpu=0
# decoding_tag="_decoding_v1"
# #######this should have at maximum number of files to decode if you want to decode all the file then this should be length of lines in scps
# max_jobs_to_decode=400 
# mem_req_decoding=10G

# log_path="$model_dir"/decoding_log_$decoding_tag
# mkdir -pv "$log_path"



# /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/queue.pl \
# 	--max-jobs-run $max_jobs_to_decode \
# 	-q short.q@@stable,short.q@@blade \
# 	--mem $mem_req_decoding \
# 	-l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
# 	JOB=1:$max_jobs_to_decode \
# 	-l 'h=!blade063' \
# 	$log_path/decoding_job.JOB.log \
# 	python TIMIT_Att_V1_LSTMSS_decoding_v2.py \
# 	--gpu $gpu \
# 	--model_dir $model_dir \
# 	--Decoding_job_no JOB


###---------------------------------------------------------------------------------------------






















#----------------------------------------------------------------------------------
#gpu=0
#decoding_tag=""
#max_jobs_to_decode=400
#mem_req_decoding=10G

#for test_folder in $dev_path $test_path
#do
#testing_folder="${test_folder##*/}"
#log_path="$PPATH"/"$model_dir"/decoding_log_$testing_folder
#mkdir -pv "$log_path"

#queue.pl --max-jobs-run $max_jobs_to_decode -q short.q@@stable,short.q@@blade --mem $mem_req_decoding -l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G JOB=1:$max_jobs_to_decode -l 'h=!blade063' $log_path/decoding_job.JOB.log python TIMIT_Att_V1_LSTMSS_decoding.py --gpu $gpu --model_dir $model_dir --Decoding_job_no JOB --dev_path $test_folder
#done






# #stdbuf -o0  python TIMIT_Att_V1_LSTMSS_v2.py --gpu $gpu

# decoding_tag=""
# max_jobs_to_decode=1000
# mem_req_decoding=10G
# log_path="$PPATH"/"$model_dir"/decoding_log_$decoding_tag
# mkdir -pv "$log_path"


# queue.pl --max-jobs-run $max_jobs_to_decode -q short.q@@stable,short.q@@blade --mem $mem_req_decoding -l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G JOB=1:$max_jobs_to_decode -l 'h=!blade063' $log_path/decoding_job.JOB.log python decoding_job.JOB.scp $beam $use_gpu $gamma $acm_wt $len_pen $mt_beam $mt_len_pen $asr_mul_mod $mt_mul_mod









 

