# Eesen

**Eesen** is a toolkit to build speech recognition (ASR) systems in a **completely end-to-end fashion**. The goal of Eesen is to **simplify** the existing complicated, expertise-intensive ASR pipeline into a straightforward learning problem. Acoustic modeling in Eesen involves training **a single recurrent neural network** (RNN) which models the sequence-to-sequence mapping from speech to transcripts. Eesen **discards the following elements** required by the existing ASR pipeline:

* Hidden Markov models (HMMs)
* Gaussian mixture models (GMMs)
* Decision trees and phonetic questions
* Dictionary, if characters are used as the modeling units
* **...**

Eesen is developed on the basis of the popular [Kaldi](http://kaldi.sourceforge.net/) toolkit. However, Eesen is fully self-contained, requiring no dependencies from Kaldi to funciton. 

Eesen is released as **an open-source project** under the highly non-restrictive **Apache License Version 2.0**. We **welcome community contribution**.

Eesen was created and has been maintained by [Yajie Miao](http://www.cs.cmu.edu/~ymiao/) from Carnegie Mellon University. Email [Yajie Miao](mailto:yajiemiao@gmail.com) if you have any questions or suggestions. 

# Recent Updates/News

* On **Switchboard**, Eesen gets the **WER% of 15.1% on Hub5'00-SWBD**, and 27.4% on Hub5'00. [[recipe](https://github.com/yajiemiao/eesen/tree/master/asr_egs/swbd/v1)]
* By directly modeling **Madarin characters**, Eesen gets the **CER% of 35.47%** on the HKUST Mandarin Telephone Speech corpus. This is a better number than reported by [the Kaldi systems](https://github.com/kaldi-asr/kaldi/blob/master/egs/hkust/s5/RESULTS). [[recipe](https://github.com/yajiemiao/eesen/tree/master/asr_egs/hkust/v1)]

<!-- Refer to [here](https://github.com/yajiemiao/eesen/wiki/Updates) for the full list of updates. -->

# References

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
* Fully-fledged [example setups](https://github.com/yajiemiao/eesen/tree/master/asr_egs) to demonstrate end-to-end system building, with both phonemes and characters as labels.

# Experimental Results

Refer to RESULTS under each [example setup](https://github.com/yajiemiao/eesen/tree/master/asr_egs).

# To-Do List

* Example setups on more datasets.
* More Wiki documentation.
* Investigate Eesen's performance on different languages and conditions (noisy, far-field, etc.).
* Accelerate model training using better learning techniques or multi-GPU distributed learning.
* **...**

