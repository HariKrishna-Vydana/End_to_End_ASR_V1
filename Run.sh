#! /bin/sh
#
#$ -q long.q@supergpu*,long.q@facegpu*,long.q@pc*,long.q@dellgpu*
#$ -l gpu=1,gpu_ram=7G,ram_free=7G,matylda3=0.5

#$ -o /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_LAS_LOC_ci_2_3_320_lrth10_WN_ls0.1_TF0.6_Normloss_lr0.0005_NORMALIZED_clipgrad5_check3_conv_frontend_inpnorm.log
#$ -e /mnt/matylda3/vydana/HOW2_EXP/Timit/log/TIMIT_LAS_LOC_ci_2_3_320_lrth10_WN_ls0.1_TF0.6_Normloss_lr0.0005_NORMALIZED_clipgrad5_check3_conv_frontend_inpnorm.log


PPATH="/mnt/matylda3/vydana/HOW2_EXP/Timit"
cd "$PPATH"
export PYTHONUNBUFFERED=TRUE


###bash variables
only_scoring='True'
scoring_path='/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/scoring/'
stage=2
#------------------------

gpu=1
max_batch_len=10
tr_disp=50
vl_disp=10
validate_interval=500
max_val_examples=400

compute_ctc=0
ctc_weight=0

learning_rate=0.0005
early_stopping=1   
clip_grad_norm=5

hidden_size=320
input_size=249
encoder_layers=3
lstm_dropout=0.3
kernel_size=3
stride=2
in_channels=1
out_channels=64
conv_dropout=0.3
isresidual=1
label_smoothing=0.1


#Attention type: LAS|Collin_monotonc|Location_aware|LAS_LOC|LAS_LOC_ci
attention_type='LAS_LOC_ci'
######'Subsamp_lstm|conv2d|nothing'
enc_front_end='conv2d'
lr_redut_st_th=10
teacher_force=0.6

min_F_bands=5
max_F_bands=30
time_drop_max=2
time_window_max=1

weight_noise_flag=1
reduce_learning_rate_flag=1
spec_aug_flag=0

pre_trained_weight="0"
#pre_trained_weight="0"
plot_fig_validation=0
plot_fig_training=0
start_decoding=1

#---------------------------
Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
Char_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'


#text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
text_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_39phseq'
train_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/'

dev_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/dev/'
test_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/test/'


model_file="TIMIT_LAS_LOC_ci_2_3_320_lrth10_WN_ls0.1_TF0.6_Normloss_lr0.0005_NORMALIZED_clipgrad5_check3_conv_frontend_inpnorm"
model_dir="$PPATH/models/$model_file"
weight_text_file="$PPATH/weight_files/$model_file"
Res_text_file="$PPATH/weight_files/$model_file"_Res


data_dir="$PPATH/Timit_training_Data_249_scps/"

mkdir -pv $model_dir
output_file="$PPATH/log/$model_file".log
log_file="$PPATH/log/$model_file".log
if [[ ! -w $weight_text_file ]]; then touch $weight_text_file; fi
if [[ ! -w $Res_text_file ]]; then touch $Res_text_file; fi
echo "$model_dir"
echo "$weight_file"
echo "$Res_file"


if [ $stage -le 1 ]; then
# #---------------------------------------------------------------------------------------------

##### making the data preperation for the experiment

stdbuf -o0  python /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/github_V1/End_to_End_ASR_V1/Make_training_scps.py \
						--data_dir $data_dir \
 						--text_file $text_file \
 						--train_path $train_path \
 						--dev_path $dev_path \
 						--Word_model_path $Word_model_path \
 						--Char_model_path $Char_model_path
##---------------------------------------------------------------------------------------------
fi
###scp wrd char


if [ $stage -le 2 ]; then
# #---------------------------------------------------------------------------------------------
##### 
echo "Training started----------:>"
stdbuf -o0  python /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/github_V1/End_to_End_ASR_V1/Training_V2.py \
 						--gpu $gpu \
 						--data_dir $data_dir\
 						--text_file $text_file\
 						--Word_model_path $Word_model_path \
 						--Char_model_path $Char_model_path \
 						--max_batch_len $max_batch_len \
 						--tr_disp $tr_disp \
 						--validate_interval $validate_interval \
 						--hidden_size $hidden_size \
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
                                                --enc_front_end $enc_front_end
##---------------------------------------------------------------------------------------------
fi



if [ $stage -le 3 ]; 
then
gpu=0
#######this should have at maximum number of files to decode if you want to decode all the file then this should be length of lines in scps
max_jobs_to_decode=400 
mem_req_decoding=10G

for test_fol in $dev_path $test_path
do
D_path=${test_fol%*/}
D_path=${D_path##*/}
echo "$test_fol"
echo "$D_path"
for beam in 10
do 
decoding_tag="_decoding_v1_beam_$beam""_$D_path"
log_path="$model_dir"/decoding_log_$decoding_tag
echo "$log_path"

mkdir -pv "$log_path"
mkdir -pv "$log_path/scoring"
mkdir -pv "$model_dir/decoding_files/plots"


if [ $only_scoring != 'True' ]; 
then

/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/utils/queue.pl \
	--max-jobs-run $max_jobs_to_decode \
	-q short.q@@stable,short.q@@blade \
	--mem $mem_req_decoding \
	-l matylda3=0.01,ram_free=$mem_req_decoding,tmp_free=10G \
	JOB=1:$max_jobs_to_decode \
	-l 'h=!blade063' \
	$log_path/decoding_job.JOB.log \
	python /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/github_V1/End_to_End_ASR_V1/Decoding.py \
	--gpu $gpu \
	--model_dir $model_dir \
	--Decoding_job_no JOB \
	--beam $beam \
	--dev_path $test_fol \
	--weight_text_file $weight_text_file\
	--Res_text_file $Res_text_file\
	--text_file $text_file

. /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file


cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $3, "(" $2 ")"}'| tr -s " "> $log_path/scoring/hyp_val_file_sc
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $4, "(" $2 ")"}'| tr -s " "> $log_path/scoring/ref_val_file_sc

bash $scoring_path/comput_wer_sclite.sh "$log_path/scoring/hyp_val_file" "$log_path/scoring/ref_val_file" "$log_path/scoring" "$log_path/scoring/hyp_val_file_sc" "$log_path/scoring/ref_val_file_sc"
cat $log_path/scoring/wer_val

else
. /mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1/path.sh
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $3}'| tr -s " "> $log_path/scoring/hyp_val_file
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $2, $4}'| tr -s " "> $log_path/scoring/ref_val_file

cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $3, "(" $2 ")"}'| tr -s " "> $log_path/scoring/hyp_val_file_sc
cat $log_path/decoding_job*.log|grep "nbest"|awk -F ' = ' '{print $4, "(" $2 ")"}'| tr -s " "> $log_path/scoring/ref_val_file_sc

bash $scoring_path/comput_wer_sclite.sh "$log_path/scoring/hyp_val_file" "$log_path/scoring/ref_val_file" "$log_path/scoring" "$log_path/scoring/hyp_val_file_sc" "$log_path/scoring/ref_val_file_sc"
cat $log_path/scoring/wer_val
fi

done
done
fi
