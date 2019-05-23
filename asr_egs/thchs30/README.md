The recipe is used for Chinese ASR with thchs30 corpus
===
## 1 Function:
	1)It can be used for Chinses ASR 

	2)You can use your own Chinese corpus to study ASR
	

## 2 Algorithm:BiLSTM+CTC+WFST  

	1)BiLSTM: 3 layers+ 1 projection layer,320 hidden units  

	2)CTC: 216 Chinese Sound finals labels + one blank label  

	3)WFST: CTC token fst(T.fst), lexicon fst(L.fst), language model fst(G.fst) 


## 3 Details: 

	1) Data preparation:  

		languange model is in data/language_model 

		lexicon.txt is in data/dict  

		Train data is in corpus/train. The format is wav+text  

		Test data is in corpus/test. The format is wav+text  

		Dev data is in corpus/dev. The format is wav+text  

	2) How to run: ./run.sh

		make_TLG_WFST.sh: it is used for generating TLG.fst .The related directory is data/{train,test,dev,lang,search_Graph}.  

		feature.sh: it is used for generating wav features. The files related wav features arein data/{train,test,dev} ,fbank  

		train.sh: Training acoustic model  

		decode.sh: Decoding with acoustic model and TLG.fst

