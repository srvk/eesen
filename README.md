# Eesen

Eesen is a framework and open-source toolkit to build end-to-end speech recognition (ASR) systems. The goal of Eesen is to simplify the existing complicated ASR pipeline into a sequence-to-sequence learning problem. Eesen is a spin-off of the popular [Kaldi](http://kaldi.sourceforge.net/) toolkit. However, it is fully self-contained, requiring no dependencies from Kaldi to funciton. Eesen has been developed by [Yajie Miao](http://www.cs.cmu.edu/~ymiao) from Carnegie Mellon University. 

# Key Components

In Eesen, end-to-end ASR is enabled by the following 3 key components
* *Acoustic Model* -- Deep RNNs with LSTM units. We apply bi-directional LSTM networks as acoustic models.
* *Training*       -- [Connectionist temporal classification (CTC)](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) as the training objective.
* *Decoding*       -- A principled decoding approach based on Weighted Finite-State Transducers (WFST). Achieves effective and efficient incorporation of language models (LMs) and lexicons during decoding. 

# Highlights

* An open-sourced toolkit with Apache License Version 2.0
* GPU implementation of LSTM model training and CTC learning; processes multiple utterances at a time in parallel
* Inherits Kaldi's programming style and reuses Kaldi's functionalities such as (feature processing and WFST wrappers). 
* Fully-fledged [example setups](https://github.com/yajiemiao/eesen/tree/master/asr_egs)

# Experimental Results

Refer to RESULTS under each example setup.

# To-Do List (short-term)

* Create TIMIT and Switchboard example setups.
* Add lattice-based decoding to example setups.
* Add Wiki pages, especially about training and decoding commands.

# To-Do List (long-term)

* Further improve Eesen's ASR accuracy from various aspects; make it eventually better than the existing hybrid DNN ASR system
* Apply the framework to more languages and speech conditions (noisy, far-field); investigate how Eesen works under these conditions.
* Speed up training by adopting better learning techniques or multi-GPU distributed learning.

# Contact

Email [Yajie Miao](mailto:yajiemiao@gmail.com) if you have any questions or suggestions.

