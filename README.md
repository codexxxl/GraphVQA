**Explainable GQA**: Explainable and Transparent GQA framework 
========


Model checkpoint on Feb 20, 2021: https://drive.google.com/drive/folders/1zHuUG-qOfOX93iB3e3WgmcAgEfFK3Dru?usp=sharing


PyTorch training code and pretrained models for Explainable GQA. 

This repo is built from scratch. 

This is the folder for phase 1 development. 
Assuming that the scene graph is give. 
The repo contains 4 modules: semantic parser, scene graph encoding, neural execution module, natural language generation module. 


# Model files:
1. simple GCN: pipeline_model_gcn.py, mainExplain_gcn.py
2. Recurrent GCN: pipeline_model.py, mainExplain.py
3. GAT: gat.py(version 1), gat_skip.py(version 2), pipeline_model_gat.py mainExplain_gat.py
4. LGRAN: lcgn.py, mainExplain_lcgn.py, pipeline_model_lcgn.py


# Install torchtext, spacy
```
conda install -c pytorch torchtext
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk
```
also need to python and run: 


```
import nltk
nltk.download('wordnet')
```

# Install PyTorch Geometric
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries


## My command with torch-1.4.0+cu100. Replace with your versions. 
```
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric
```




# Prepare data

Download scene graphs: https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
Download questions: https://nlp.stanford.edu/data/gqa/questions1.2.zip


put sceneGraph json files into sceneGraphs/
put questions json files into questions/original/ 


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


# Fix Data Path
<!-- modify 
ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/') in the following 3 files, Constants.py, gqa_dataset_entry.py, preprocess.py

replace with your own root path. Here my folder is '/home/weixin/neuralPoolTest/explainableGQA' so I use '/home/weixin/neuralPoolTest' without 'explainableGQA'

For the file gqa_dataset_entry.py, replace two additional paths: SCENEGRAPHS and EXPLAINABLE_GQA_DIR with your own sceneGraphs and explainableGQA folder paths. -->


modify 
ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/') in the file Constants.py.

replace with your own root path. Here my folder is '/home/weixin/neuralPoolTest/explainableGQA' so I use '/home/weixin/neuralPoolTest' without 'explainableGQA'

# Preprocess Question files:
```
python preprocess.py
```


# Testing the installation
```
python pipeline_model.py 
python gqa_dataset_entry.py 
```



# Train
```
Single GPU training: 
CUDA_VISIBLE_DEVICES=2 python mainExplain.py --log-name debug.log && echo 'Ground Truth Scene Graph Debug'

Distributed Training:
CUDA_VISIBLE_DEVICES=0,1,2,7 python -m torch.distributed.launch --nproc_per_node=4 --use_env mainExplain.py

Kill Distributed:
kill $(ps aux | grep mainExplain.py | grep -v grep | awk '{print $2}')

For all models:
1. simple GCN: run mainExplain_gcn.py instead
2. Recurrent GCN: run mainExplain.py instead
3. GAT: run mainExplain_gat.py instead
4. LGRAN: run mainExplain_lcgn.py instead
```


# Bonus 1: Run the LCGN baseline, 90.23% Accuracy on val_balanced


https://github.com/ronghanghu/lcgn/tree/pytorch


* R. Hu, A. Rohrbach, T. Darrell, K. Saenko, *Language-Conditioned Graph Networks for Relational Reasoning*. in ICCV 2019 ([PDF](https://arxiv.org/pdf/1905.04405.pdf))
```
@inproceedings{hu2019language,
  title={Language-Conditioned Graph Networks for Relational Reasoning},
  author={Hu, Ronghang and Rohrbach, Anna and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

Project Page: http://ronghanghu.com/lcgn

**This is the (original) TensorFlow implementation of LCGN. A PyTorch implementation is available in the [PyTorch branch](https://github.com/ronghanghu/lcgn/tree/pytorch).**



# Bonus 2: Run the Symbolic Execution baseline
WACV 2021 Paper "Meta Module Network for Compositional Visual Reasoning"


The symbolic execution for the visual question answering.


## Symbolic Execution
We can run the run.py to perform symbolic execution on the GQA provided scene graph to get the answer.
  ```
    python run.py --do_trainval_unbiased
  ```
The script will return 
  ```
  success rate (ALL) = 0.923610917323838, success rate (VALID) = 0.9605288360151759, valid/invalid = 1033742/41320
  ```
It means that for those questions, whose answer is inside the scene graph, the accuracy is 96%. There are 4% of questions without answers inside the scene graph, therefore the overall accuracy is 92.3%. This is good enough as a symbolic teacher to teach the meta module network to reason.


## Meta Data Sources
https://github.com/microsoft/DFOL-VQA

http://ronghanghu.com/lcgn

https://github.com/wenhuchen/Meta-Module-Network
