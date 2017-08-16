### Eesen

**Eesen** is to simplify the existing complicated, expertise-intensive ASR pipeline into a straightforward learning problem. Acoustic modeling in Eesen involves training a single recurrent neural network (RNN) to model the mapping from speech to transcripts. Eesen discards the following elements required by the existing ASR pipeline:

* Hidden Markov models (HMMs)
* Gaussian mixture models (GMMs)
* Decision trees and phonetic questions
* Dictionary, if characters are used as the modeling units
* **...**

Eesen was created by [Yajie Miao](http://www.cs.cmu.edu/~ymiao) based on the [Kaldi](http://kaldi.sourceforge.net/) toolkit.

### Key Components

Eesen contains 3 key components to enable end-to-end ASR:
* **Acoustic Model** -- Bi-directional RNNs with LSTM units.
* **Training**       -- [Connectionist temporal classification (CTC)](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) as the training objective.
* **Decoding**       -- A principled decoding approach based on Weighted Finite-State Transducers (WFSTs).  

### Highlights of Eesen

* The WFST-based decoding approach can incorporate lexicons and language models into CTC decoding in an effective and efficient way. 
* GPU implementation of LSTM model training and CTC learning.
* Multiple utterances are processed in parallel for training speed-up.
* Fully-fledged [example setups](asr_egs/) to demonstrate end-to-end system building, with both phonemes and characters as labels.

### Experimental Results

Refer to RESULTS under each [example setup](asr_egs/).


### References

For more information, please refer to the following paper(s):

Yajie Miao, Mohammad Gowayyed, and Florian Metze, "[EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding](http://arxiv.org/abs/1507.08240)," in Proc. ASRU 2015.

For max perturbation and dropout:

Jayadev Billa, "[Improving LSTM-CTC based ASR performance in domains with limited training data](https://arxiv.org/abs/1707.00722)"
