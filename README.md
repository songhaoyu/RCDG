# Generating Persona Consistent Dialogues by Exploiting Natural Language Inference

Source code for RCDG model from AAAI20 *Generating Persona Consistent Dialogues by Exploiting Natural Language Inference*.


## Requirements:

The code is tested under the following env:

* Python 3.6
* Pytorch 0.3.1

Install with conda: `conda install pytorch==0.3.1 torchvision cudatoolkit=7.5 -c pytorch`

This released code has been tested on a Titan-XP 12G GPU.

## Data
We have provided some data samples in `./data` to show the format. For downloading the full datasets, please refer to the following papers:

* **PersonaChat:** https://www.aclweb.org/anthology/P18-1205/

* **DNLI:** https://www.aclweb.org/anthology/P19-1363/

## How to Run:

For a easier way to run the code, here the NLI model is GRU+MLP, i.e. RCDG_base, and we remove the time-consuming MC search. 

Here are a few steps to run this code:


### 0. Prepare Data
```
python preprocess.py -train_src data/src-train.txt -train_tgt data/tgt-train.txt -train_per data/per-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -valid_per data/per-val.txt -train_nli data/nli-train.txt -valid_nli data/nli-valid.txt -save_data data/nli_persona -src_vocab_size 18300 -tgt_vocab_size 18300 -share_vocab
```


And as introduced in the paper, there are different training stages:

### 1. NLI model Pretrain

`cd NLI_pretrain/`

```
python train.py -data ../data/nli_persona -batch_size 32 -save_model saved_model/consistent_dialogue -rnn_size 500 -word_vec_size 300 -dropout 0.2 -epochs 5 -learning_rate_decay 1 -gpu 0
```

And you should see something like:

```
Loading train dataset from ../data/nli_persona.train.1.pt, number of examples: 1
31432
Epoch  1, nli_step     1/ 4108; nli: 0.28125
Epoch  1, nli_step    11/ 4108; nli: 0.38125
Epoch  1, nli_step    21/ 4108; nli: 0.43438
Epoch  1, nli_step    31/ 4108; nli: 0.48125
Epoch  1, nli_step    41/ 4108; nli: 0.53750
Epoch  1, nli_step    51/ 4108; nli: 0.56250
Epoch  1, nli_step    61/ 4108; nli: 0.49062
...
```

### 2. Generator G Pretrain

`cd ../G_pretrain/`

```
python train.py -data ../data/nli_persona -batch_size 32 -rnn_size 500 -word_vec_size 300  -dropout 0.2 -epochs 15 -g_optim adam -g_learning_rate 1e-3 -learning_rate_decay 1 -train_from PATH_TO_PRETRAINED_NLI -gpu 0
```
Here the `PATH_TO_PRETRAINED_NLI` should be replaced by your model path, e.g., `../NLI_pretrain/saved_model/consistent_dialogue_e3.pt`.

If , you should see the ppl comes down during training, which means the dialogue model is in training:

```
Loading train dataset from ../data/nli_persona.train.1.pt, number of examples: 131432
Epoch  4, teacher_force     1/ 4108; acc:   0.00; ppl: 18619.76; 125 src tok/s; 162 tgt tok/s;      3 s elapsed
Epoch  4, teacher_force    11/ 4108; acc:   9.69; ppl: 2816.01; 4159 src tok/s; 5468 tgt tok/s;      3 s elapsed
Epoch  4, teacher_force    21/ 4108; acc:   9.78; ppl: 550.46; 5532 src tok/s; 6116 tgt tok/s;      4 s elapsed
Epoch  4, teacher_force    31/ 4108; acc:  11.15; ppl: 383.06; 5810 src tok/s; 6263 tgt tok/s;      5 s elapsed
...
Epoch  4, teacher_force   941/ 4108; acc:  25.40; ppl:  90.18; 5993 src tok/s; 6645 tgt tok/s;     63 s elapsed
Epoch  4, teacher_force   951/ 4108; acc:  27.49; ppl:  77.07; 5861 src tok/s; 6479 tgt tok/s;     64 s elapsed
Epoch  4, teacher_force   961/ 4108; acc:  26.24; ppl:  83.17; 5473 src tok/s; 6443 tgt tok/s;     64 s elapsed
Epoch  4, teacher_force   971/ 4108; acc:  24.33; ppl:  97.14; 5614 src tok/s; 6685 tgt tok/s;     65 s elapsed
...
```

### 3. Discriminator D Pretrain
`cd ../D_pretrain/`

```
python train.py -epochs 20 -d_optim adam -d_learning_rate 1e-4 -data ../data/nli_persona -train_from PATH_TO_PRETRAINED_G -batch_size 32 -learning_rate_decay 0.99 -gpu 0
```

Similarly, replace `PATH_TO_PRETRAINED_G ` with the `G Pretrain` model path.

The acc of D will be displayed during training:

```
Loading train dataset from ../data/nli_persona.train.1.pt, number of examples: 131432
Epoch  5, d_step     1/ 4108; d: 0.49587
Epoch  5, d_step    11/ 4108; d: 0.51580
Epoch  5, d_step    21/ 4108; d: 0.49853
Epoch  5, d_step    31/ 4108; d: 0.55248
Epoch  5, d_step    41/ 4108; d: 0.55168
...
```

### 4. Adversarial Reinforcement Training

`cd ../reinforcement_train/`

```
python train.py -epochs 30 -batch_size 32 -d_learning_rate 1e-4 -g_learning_rate 1e-4 -learning_rate_decay 0.9 -data ../data/nli_persona -train_from PATH_TO_PRETRAINED_D -gpu 0
```

Remember to replace `PATH_TO_PRETRAINED_D ` with the `D Pretrain` model path.

Note that all the `-epochs` are global among all stages,  if you want to tune this parameter. Actually, there are 30 - 20 = 10 training epochs in this Adversarial Reinforcement Training stage if the D Pretrain model was trained 20 epochs in total.
 

###5. Testing Trained Model
Now we have a trained dialogue model, we can test by:

Still in `./reinforcement_train/`


```
python predict.py -model  -src ../data/src-val.txt -tgt ../data/tgt-val.txt -replace_unk -verbose -output ./results.txt -per ../data/per-val.txt -nli nli-val.txt -gpu 0
```

## MISC
* **Why the dependencies look so outdated?**	
	The project was finished in early 2019, though just releasted. Due to the rapid updates of Pytorch and OpenNMT, running this project with their subsequent versions will cause unexpeted compatibility problems.
	
* **Initializing Model Seems Slow?**	
	
	This is a legacy problem due to pytorch < 0.4, not brought by this project. And the training efficiency will not be affected.


* **BibTex**
	
	```
	@article{Song_RCDG_2020,
		title={Generating Persona Consistent Dialogues by Exploiting Natural Language Inference},
		volume={34},
		DOI={10.1609/aaai.v34i05.6417},
		number={05},
		journal={Proceedings of the AAAI Conference on Artificial Intelligence},
		author={Song, Haoyu and Zhang, Wei-Nan and Hu, Jingwen and Liu, Ting},
		year={2020},
		month={Apr.},
		pages={8878-8885}
		}
	```
	
