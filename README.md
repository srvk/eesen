### Eesen

**Eesen** is to simplify the existing complicated, expertise-intensive ASR pipeline into a straightforward sequence learning problem. Acoustic modeling in Eesen involves training a single recurrent neural network (RNN) to model the mapping from speech to text. Eesen abandons the following elements required by the existing ASR pipeline:

* Hidden Markov models (HMMs)
* Gaussian mixture models (GMMs)
* Decision trees and phonetic questions
* Dictionary, if characters are used as the modeling units
* **...**

Eesen was created by [Yajie Miao](http://www.cs.cmu.edu/~ymiao) borrowing liberally from the [Kaldi](https://github.com/kaldi-asr/kaldi) toolkit. [Thank you, Yajie!](https://www.youcaring.com/iscainternationalspeechcommunicationassociation-815026)

### Key Components

Eesen contains 4 key components to enable end-to-end ASR:
* **Acoustic Model**  -- Bi-directional RNNs with LSTM units.
* **Training**        -- [Connectionist temporal classification (CTC)](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) as the training objective.
* **WFST Decoding**   -- A principled decoding approach based on Weighted Finite-State Transducers (WFSTs), or 
* **RNN-LM Decoding** -- Decoding based on (character) [RNN language models](https://arxiv.org/abs/1408.2873), when using Tensorflow

### Highlights of Eesen

* The WFST-based decoding approach can incorporate lexicons and language models into CTC decoding in an effective and efficient way.
* The RNN-LM decoding approach does not require a fixed lexicon.
* GPU implementation of LSTM model training and CTC learning, now also using [Tensorflow](https://www.tensorflow.org/).
* Multiple utterances are processed in parallel for training speed-up.
* Fully-fledged [example setups](asr_egs/) to demonstrate end-to-end system building, with both phonemes and characters as labels, following [Kaldi](https://github.com/kaldi-asr/kaldi) conventions.

### Experimental Results

Refer to RESULTS under each [example setup](asr_egs/).

### References

For more information, please refer to the following paper(s):

Yajie Miao, Mohammad Gowayyed, and Florian Metze, "[EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding](http://arxiv.org/abs/1507.08240)," in Proc. Automatic Speech Recognition and Understanding Workshop (ASRU), Scottsdale, AZ; U.S.A., December 2015. IEEE.
