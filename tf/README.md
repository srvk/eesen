README for Tensorflow
--

**rnn-lm** contains the code to train a character lm in Tensorflow 

**ctc-decode** contains Thomas' code for ctc decoding with an RNN LM

**ctc-train** contains the latest code for training (the classic eesen code, ported by Bo)

* help: 
```
#!bash
python main.py --help

```

* train:
```
#!bash
python main.py --use_cudnn --store_model

```

* Evaluate:

```
#!bash
python main.py --eval --eval_model=/home/bchen2/Haitian/log/dbr-run4/model/epoch30.ckpt

```

* Tensorboard
```
#!bash
tensorboard --logdir=Haitian/log/
```
and then use port forwarding with ssh such as 
```
#!bash
ssh -L 6006:localhost:6006 $rocks
```
and open local browser, and visit *localhost:6006*
