#!/usr/bin/python
import sys                                                                  
                                                                                                        
def Init_model_classes(args):
        """ In case if we have multiple scripts selecting them, 
        for now there is only one encoder and decoder scripts 
        """

        sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/Gen_V1/ATTNCODE/Basic_Attention_V1')
        from Res_LSTM_Encoder_arg import Conv_Res_LSTM_Encoder as encoder
        from Decoder_V1 import decoder

        model_encoder=encoder(args=args)
        model_decoder=decoder(args=args)
        
        return model_encoder,model_decoder
