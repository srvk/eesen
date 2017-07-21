
This dir contains fully fledged recipes to build end-to-end ASR systems using
the Wall Street Journal (WSJ) corpus.

You need to obtain the WSJ dataset from LDC to run this example. The LDC catalog
numbers are LDC93S6B and LDC94S13B. 

There are two recipes, demonstrating different types of CTC labels

run_ctc_phn.sh   - phonemes as CTC labels   
run_ctc_char.sh  - characters (letters) as CTC labels

Please note that you need IRSTLM to run the current version of the recipe without changes.

