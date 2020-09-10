# End_to_End_ASR_V1

....edit it later

#General Training procedure:
1. All the general and default parameters for training and decoding are in Attention_arg.py, they can be changed by adding them in the bash script. The model ASR and RNNLM parses the argumnets and writes the final paremeters as a jsom dump in the path (EXP="models/"present_exp"/model_architecture_") this will be later accessed by the decoding script to load the model and decode the models. 
2. Initially train the sentence-piece model to get one of the word, char, bpe, unigram models. These models give you the vocablary size.
3. The dataloader takes text and makes a dictionary of text_dict={utt_id : transcription (as a string)} (need ram to support this dict storage).
4. Dataloder iterates over the scp files and for each line in scp file picks the utt_id and gets the corrsponding transcription from the text_dict. The feat,text are stored in lists and when the datasize grows more than the max_batch_len(No of frames in the batch) or max_batch_size(no of utteracnes in the batch) the Dataloader ouputs the batch to queue in the form of a dict


## Data loading and pre-proceesing
5. Dataloading is done by Dataloader_for_AM_v1.py, which outputs a data_dict

6. Training_loop.py acces the data dictnary by name and converts to pytorch and uses it

	-->has SpecAugment on numpy matrices 
		Needs to set the  (spec_aug_flag=True) in the argparse and the when the model starts to overfit i.e., when the learning rate gets reduced from the initial value the Spec_Aug will be swithed on with rmentioned arguments.

	-->weight noise regularization step
		Operates similar to Spec_Aug and plays a crucial role while training on data with smaler sizes.



## model initialization and training
7. Initializing_model_LSTM_SS_v2_args.py initilizes the model and moves it to gpu, if args.gpu is true
	: there is scope for adding new modules in future (Transformer ASR) 
8. CE_loss_label_smoothiong.py has the Negitive loglikelhod loss with label smoothing
9. user_defined_losses.py  has edit distance metric for traking the earlystopping



## Decoding from the model
10. Decoding.py loads the model and rnnlm model. The Decoding.py searches for the file in the ( folder "EXP") and initilizes the model and loads the best retrained weights.
11. reads the lofg files from weight_files/"present_exp" TER from weight_files/"present_exp_Res" and gets the best weight based on the WER and sets the weight as pretrained_weight and model is initialized with these weights.
11. Decoding_loop.py takes scp files and writes transcription and WER.
