
This dir contains recipes to build Eesen systems on Switchboard.

At the minimum, you need to obtain the following LDC datasets:
LDC97S62  LDC2002S09   LDC2002T43

[Optional] To build LMs with the Fisher transcripts, you need 2 additional datasets:
LDC2004T19   LDC2005T19

There are two recipes, demonstrating different types of CTC labels
run_ctc_phn.sh   - phonemes as CTC labels   
run_ctc_char.sh  - characters (letters) as CTC labels

