# Distance Parser
Distance parser is a supervised constituency parser based on syntactic distance. 
This repo is a working sample of distance parser which reproduces the results reported in the paper 
[Straight to the Tree: Constituency Parsing with Neural Syntactic Distance](https://arxiv.org/abs/1806.04168), 
which is published in ACL 2018. We provide models with proper configurations for PTB and CTB datasets, as well as their preprocessing scripts.

## Requirements
[PyTorch](https://pytorch.org/) We use PyTorch 0.4.0 with python 3.6.   
[Stanford POS tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-02-27.zip). We use the full Stanford Tagger, version 3.9.1, build 2018-02-27.   
[NLTK](http://www.nltk.org/) We use NLTK 3.2.5.  
[EVALB](https://nlp.cs.nyu.edu/evalb/) We have integrated a compiled EVALB inside our repo. This compiled version is forked from the current latest verison of EVALB, which can be accessed through [this link](https://nlp.cs.nyu.edu/evalb/EVALB.tgz).  

## Datasets and Preprocessing

### Preprocessing PTB
We use the same preprocessed PTB files from the [self attentive parser](https://github.com/nikitakit/self-attentive-parser) repo. [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) are optional if you don't want to run the ablation experiments.

To preprocess PTB, please follow the steps below:

1. Download the 3 PTB data files from https://github.com/nikitakit/self-attentive-parser/tree/master/data, and put them in the `data/ptb` folder.

2. Run the following command to prepare the PTB data:
```
python datacreate_ptb.py ../data/ptb /path/to/glove.840B.300d.txt
```

### Preprocessing CTB
We use the standard train/valid/test split specified in [Liu and Zhang, 2017](https://arxiv.org/pdf/1707.05000.pdf) for our CTB experiments.

To preprocess the CTB, please follow the steps below:

1. Download and unzip the Chinese Treebank dataset from https://wakespace.lib.wfu.edu/handle/10339/39379

2. If you don't have any data in NLTK before, download some to initialize your `nltk_data` folder:
```
python -c "import nltk; nltk.download('ptb')"
``` 

3. Run the following command to link the dataset to NLTK, and generate the train/valid/test split in the repo:
```
python ctb.py --ctb /path/to/your/ctb8.0/data --output data/ctb_liusplit
```

4. Integrate the Stanford Tagger for data preprocessing. Download the Stanford tagger from https://nlp.stanford.edu/software/stanford-postagger-full-2018-02-27.zip, and edit the lines 10~11 in file `src/datacreate_ctb.py` to your tagger path.

5. Run the following command to generate the preprocessed files:
```
python datacreate_ctb.py ../data/ctb_liusplit /pth/to/stanford/tagger/
```

## Experiments
For reproducing the PTB results in table 1, run 
```
cd src
python dp.py --cuda --datapath ../data/ptb --savepath ../ptbresults --epc 200 --lr 0.001 --bthsz 20 --hidsz 1200 --embedsz 400 --window_size 2 --dpout 0.3 --dpoute 0.1 --dpoutr 0.2 --weight_decay 1e-6
```

For reproducing the CTB results in table 2, run 
```
cd src
python dp.py --cuda --datapath ../data/ctb_liusplit --savepath ../ctbresults --epc 200 --lr 0.001 --bthsz 20 --hidsz 1200 --embedsz 400 --window_size 2 --dpout 0.4 --dpoute 0.1 --dpoutr 0.1 --weight_decay 1e-6
```

## Pre-trained models
The following pre-trained parser models are available in our repo
```
results/ptb.th  # Our best English single-system parser (92.0 F1)
results/ctb.th  # Our best Chinese single-system parser (86.5 F1)
```
To re-evaluate the pre-trained models, run:
```
python demo.py --cuda --datapath ../data/ptb/ --filename ptb
python demo.py --cuda --datapath ../data/ctb_liusplit/ --filename ctb
```
You may need [Git LFS](https://git-lfs.github.com/) to download the pre-trained models.