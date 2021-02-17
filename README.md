**Explainable GQA**: Explainable and Transparent GQA framework 
========
PyTorch training code and pretrained models for Explainable GQA. 

This repo is built from scratch. 

This is the folder for phase 1 development. 
Assuming that the scene graph is give. 
The repo contains 4 modules: semantic parser, scene graph encoding, neural execution module, natural language generation module. 



# Prepare data

Download scene graphs: https://nlp.stanford.edu/data/gqa/sceneGraphs.zip
Download questions: https://nlp.stanford.edu/data/gqa/questions1.2.zip


put sceneGraph json files into sceneGraphs/
put questions json files into questions/original/ (optional)

# Install torchtext, spacy
conda install -c pytorch torchtext
conda install -c conda-forge spacy
conda install -c conda-forge cupy
python -m spacy download en_core_web_sm
conda install -c anaconda nltk

also need to python and run: 
>>> import nltk
>>> nltk.download('wordnet')

# Install PyTorch Geometric
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-via-binaries


## My command with torch-1.4.0+cu100
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.4.0+cu100.html
pip install torch-geometric



# Fix Data Path
ROOT_DIR = pathlib.Path('/home/weixin/neuralPoolTest/')

replace with your own root path. Here my folder is '/home/weixin/neuralPoolTest/explainableGQA'



# Test code
python pipeline_model.py 
python gqa_dataset_entry.py 