# Eesen

**Eesen** is a toolkit to build speech recognition (ASR) systems in a **completely end-to-end fashion**. The goal of Eesen is to **simplify** the existing complicated, expertise-intensive ASR pipeline into a straightforward learning problem. Acoustic modeling in Eesen involves training **a single recurrent neural network** (RNN) which models the sequence-to-sequence mapping from speech to transcripts. Eesen **discards the following elements** required by the existing ASR pipeline:

* Hidden Markov models (HMMs)
* Gaussian mixture models (GMMs)
* Decision trees and phonetic questions
* Dictionary, if characters are used as the modeling units
* **...**

Eesen is developed on the basis of the popular [Kaldi](http://kaldi.sourceforge.net/) toolkit. However, Eesen is fully self-contained, requiring no dependencies from Kaldi to funciton. 

Eesen is released as **an open-source project** under the highly non-restrictive **Apache License Version 2.0**. We welcome community participation and contribution.

For **more informaiton**, please refer to our manuscript:
[EESEN: End-to-End Speech Recognition using Deep RNN Models and WFST-based Decoding](http://arxiv.org/abs/1507.08240). 

# Key Components

Eesen contains 3 key components to enable end-to-end ASR:
* **Acoustic Model** -- Bi-directional RNNs with LSTM units.
* **Training**       -- [Connectionist temporal classification (CTC)](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) as the training objective.
* **Decoding**       -- A principled decoding approach based on Weighted Finite-State Transducers (WFSTs).  

# Highlights of Eesen

* The WFST-based decoding approach can incorporate lexicons and language models into CTC decoding in an effective and efficient way. 
* GPU implementation of RNN model training and CTC learning.
* Multiple utterances are processed in parallel for training speed-up.
* Inherits Kaldi's programming stype. Convenient to implement new modules. 
* Eesen's close connection with Kaldi makes the end-to-end systems directly comparable to Kaldi's hybrid HMM/DNN systems.
* Fully-fledged [example setups](https://github.com/yajiemiao/eesen/tree/master/asr_egs) to demonstrate end-to-end system building, with both phonemes and characters as labels.

# Updates

Refer to [here](https://github.com/yajiemiao/eesen/wiki/Updates) for a list of recent updates.

# Experimental Results

Refer to RESULTS under each example setup.

# To-Do List (short-term)

* Create TIMIT example setups.
* Add CPU-based training.
* More Wiki pages/documentation, especially about training and decoding commands.

# To-Do List (long-term)

* Further improve Eesen's ASR accuracy from various aspects, to make it eventually better than the state-of-the-art hybrid HMM/DNN pipeline.
* Investigate the advantages and disadvantages of Eesen on different languages and speech conditions (noisy, far-field, etc.).
* Accelerate model training by adopting better learning techniques or multi-GPU distributed learning.

# Contact

Email [Yajie Miao](mailto:yajiemiao@gmail.com) if you have any questions or suggestions.
