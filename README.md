**Explainable GQA**: Explainable and Transparent GQA framework 
========
PyTorch training code and pretrained models for Explainable GQA. 

This repo is built from scratch. 

This is the folder for phase 1 development. 
Assuming that the scene graph is give. 
The repo contains 4 modules: semantic parser, scene graph encoding, neural execution module, natural language generation module. 



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



# Fix Data Path
modify 
ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/') in the following 3 files, Constants.py, gqa_dataset_entry.py, preprocess.py

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

```


# Bonus: Run the LCGN baseline, 90.23% Accuracy on val_balanced


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