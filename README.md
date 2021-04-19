# GraphVQA: Language-Guided Graph Neural Networks for Scene Graph Question Answering


This repo provides the source code of our paper: [GraphVQA: Language-Guided Graph Neural Networks for Scene Graph Question Answering]() (NAACL 2021).
```
@InProceedings{2021graphvqa,
  author =  {Weixin Liang and Yanhao Jiang and Antoine Bosselut and Zixuan Liu},
  title =   {GraphVQA: Language-Guided Graph Neural Networks for Scene Graph Question Answering},
  year =    {2021},  
  booktitle = {North American Chapter of the Association for Computational Linguistics (NAACL)},  
}
```


<p align="center">
  <img src="./figs/graphVQA_overview.jpg" width="1000" title="Overview of Visual Question Answering with Scene Graphs" alt="">
</p>
<p align="center">
  <img src="./figs/graphVQA_framework.jpg" width="1000" title="Structure of GraphVQA framework" alt="">
</p>


## Usage
### 0. Dependencies

create a conda environment with Python version = 3.6

#### 0.1. Install torchtext, spacy
Run following commands in created conda environment
```
conda install -c pytorch torchtext
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk
```

Excute python and run following:
```
import nltk
nltk.download('wordnet')
```

#### 0.2. Install PyTorch Geometric
Following link below to install PyTorch Geometric via binaries.
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries

Example installation commands with torch-1.4.0+cu100 are following. (Note you need to replace PyTorch and CUDA fields with your own installed versions.)
```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric
```


### 1. Download Data

Download scene graphs raw data from: 
https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
Download questions raw data from: 
https://nlp.stanford.edu/data/gqa/questions1.2.zip

Put sceneGraph json files into ```sceneGraphs/```
Put questions json files into ```questions/original/```

After this step, the file structure should look like
```
explainableGQA
    questions/
        original/
            train_balanced_questions.json
            val_balanced_questions.json
            test_balanced_questions.json
            testdev_balanced_questions.json
    sceneGraphs/
        train_sceneGraphs.json
        val_sceneGraphs.json
```



### 2. Modify Root Directory
Replace line 13 in Constants.py with your own root directory that contains this source code folder:
```ROOT_DIR = pathlib.Path('/Users/yanhaojiang/Desktop/cs224w_final/')```

For example, if my source code folder is 
`/home/weixin/neuralPoolTest/explainableGQA `
I can replace ROOT_DIR with the following path (Note without the folder name 'explainableGQA'):
```ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/')```


### 3. Preprocess Question Files (just need to run once)
run command
```
python preprocess.py
```

### 4. Test Installations and Data Preparations
Following commands should run without error:
```
python pipeline_model_gat.py 
python gqa_dataset_entry.py 
```



### 5. Training 

#### 5.1. Main Model: GraphVQA-GAT 
Single GPU training: 
```CUDA_VISIBLE_DEVICES=0 python mainExplain_gat.py --log-name debug.log ```

Distributed training:
```CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env mainExplain_gat.py --workers=4 --batch-size=200 --lr_drop=90```

To kill a distributed training:
```kill $(ps aux | grep mainExplain_gat.py | grep -v grep | awk '{print $2}')```


#### 5.2. Baseline and Test Models
Baseline and other test models are trained in similar ways with corresponding `mainExplain_{lcgn, gcn, gine}.py` file excuted. Their files are appended under folder `\baseline_and_test_models`. (Note move them out of this folder to train).
Corresponding to GraphVQA-GAT's model and training files: `gat_skip.py`, `pipeline_model_gat.py`, and `mainExplain_gat.py`, baseline model files are:

1. Baseline LCGN: `lcgn.py`, `mainExplain_lcgn.py`, `pipeline_model_lcgn.py`
2. GraphVQA-GCN: `pipeline_model_gcn.py`, `mainExplain_gcn.py`
3. GraphVQA-GINE: `pipeline_model_gine.py`, `mainExplain_gine.py`

