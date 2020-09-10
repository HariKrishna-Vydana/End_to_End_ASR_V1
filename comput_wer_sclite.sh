#!/usr/bin/bash

. ./path.sh

GT=$1
hyp=$2



/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT -h $hyp -i rm -o all stdout > result.wrd.txt

/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT -h $hyp -i rm -o dtl stdout > result.wrd.dtl

/mnt/matylda3/vydana/eesen/tools/sctk/bin/sclite -r $GT -h $hyp -i rm -o pralign stdout > result.wrd.pralign


# dtl,lur,pralign,prf,rsum,sgml,spk,snt,sum,wws 
#align-text ark,t:$GT ark,t:$hyp ark,t:alignment.txt_val
#cat alignment.txt_val|wer_per_utt_details.pl>per_utt_alignment.txt_val
#compute-wer --text --mode=present ark:$GT  ark,t:$hyp 2>&1> wer_val
