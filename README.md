# Eesen

Eesen is a framework and open-source toolkit to build end-to-end speech recognition (ASR) systems. The goal of Eesen is to simplify the existing complicated ASR pipeline into a sequence-to-sequence learning problem. Eesen is a spin-off of the popular [Kaldi](http://kaldi.sourceforge.net/) toolkit, and has been developed by [Yajie Miao](http://www.cs.cmu.edu/~ymiao) from Carnegie Mellon University. 

# Key Components

In Eesen, end-to-end ASR is enabled by the following 3 key components
* *Acoustic Model* -- Deep RNNs with LSTM units. We apply bi-directional LSTM networks as acoustic models.
* *Training*       -- [Connectionist temporal classification (CTC)](http://www.machinelearning.org/proceedings/icml2006/047_Connectionist_Tempor.pdf) as the training objective.
* Decoding       -- A principled decoding approach based on Weighted Finite-State Transducers (WFST). Achieves effective and efficient incorporation of language models (LMs) and lexicons during decoding. 

# Highlights

* An open-sourced toolkit with Apache License Version 2.0
* GPU implementation of LSTM model training and CTC learning
* Inherits Kaldi's programming style and reuses Kaldi's functionalities such as (feature processing and WFST wrappers). 
* Fully-fledged example setups
