# End_to_End_ASR_V1

....edit it later

1. Arguments are training and decoding are in Attention_arg.py

## Data loading and pre-proceesing

2. Dataloading is done by Dataloader_for_AM_v1.py, which outputs a data_dict
3. Training_loop.py acces the data dictnary by name and converts to pytorch and uses it 
	-->has SpecAugment on numpy matrices
	-->weight noise regularization step

## model initialization and training
4. Initializing_model_LSTM_SS_v2_args.py initilizes the model and moves it to gpu, if args.gpu is true
	: there is scope for adding new modules in future (Transformer ASR) 

5. CE_loss_label_smoothiong.py has the Negitive loglikelhod loss with label smoothing
6. user_defined_losses.py  has edit distance metric for traking the earlystopping


## Decodint from the model
7. Decoding.py loads the model and rnnlm model 
8. Decoding_loop.py takes scp files and writes transcription and WER



How to train...........
