# TransKG:The translation model for knowledge Graph
Knowledge Graph Embedding model collections implemented by Pytorch.Including TransE<sup>1</sup>,TransH<sup>2</sup>,TransR<sup>3</sup>,TransD<sup>4</sup>,TransA<sup>5</sup>,KG2E<sup>6</sup> models for Knowledge Graph.

Also you can choose some of old models,including NTN,SLM,LFM,SME model.

# requirements
+ tqdm>=4.59.0
+ torch>=1.8.0
+ pandas>=1.2.3
+ numpy>=1.20.1
+ tensorboardX>=2.2
# Usage
Step1: Create a virtual environment for the python.
```bash
virtualenv -p /usr/bin/python pytorch_env
```
Step2: Install the requirements.
```bash
source pytorch_env/bin/activate
```
Step3: Install the requirements.
```bash
pip install -r requirements.txt
```
Step4: Run the `main.py` script to train the model.
```bash
python main.py
```
The arguments of the file `main.py` as follows:
```bash
usage: TransKG framework for training embeddings. [-h] [--model-name MODEL_NAME] [--ent-dim ENT_DIM] [--rel-dim REL_DIM] [--dataset-name DATASET_NAME] [--shuffle SHUFFLE]
                                                  [--opt-method OPT_METHOD] [--learning-rate LEARNING_RATE] [--grad-clipping GRAD_CLIPPING] [--weight-decay WEIGHT_DECAY]
                                                  [--lr-decay LR_DECAY] [--momentum MOMENTUM] [--rho RHO] [--eps EPS] [--cuda CUDA] [--gpu GPU] [--num-workers NUM_WORKERS]
                                                  [--parallel PARALLEL] [--random-seed RANDOM_SEED] [--num-epoches NUM_EPOCHES] [--batch-size BATCH_SIZE]
                                                  [--save-steps SAVE_STEPS] [--root-dir ROOT_DIR] [--checkpoints-dir CHECKPOINTS_DIR] [--log-file LOG_FILE]
                                                  [--pre-model PRE_MODEL] [--emb-file EMB_FILE] [--checkpoint CHECKPOINT]
```
Then the embedding,model,parameters file will save in dir:`./checkpoints`
And the dictionary of entity and relation file,train,valid and test file will save in directory:`./data`.
# Dataset
The FB15K-237 dataset is here:[FB15K-237](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz).
# References
[1] [Bordes A ,  Usunier N ,  Garcia-Duran A , et al. Translating Embeddings for Modeling Multi-relational Data. Curran Associates Inc.  2013.](http://www.thespermwhale.com/jaseweston/papers/CR_paper_nips13.pdf)

[2] [Zhang J . Knowledge Graph Embedding by Translating on Hyperplanes[J]. AAAI - Association for the Advancement of Artificial Intelligence, 2015.](http://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf)

[3] [Learning Entity and Relation Embeddings for Knowledge Graph Completion Lin Y, Liu Z, Zhu X, et al. AAAI. 2015. ](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf)

[4] [Ji G ,  He S ,  Xu L , et al. Knowledge Graph Embedding via Dynamic Mapping Matrix[C]// Meeting of the Association for Computational Linguistics & the International Joint Conference on Natural Language Processing. 2015.](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Knowledge%20Graph%20Embedding%20via%20Dynamic%20Mapping%20Matrix.pdf)

[5] [Xiao H ,  Huang M ,  Hao Y , et al. TransA: An Adaptive Approach for Knowledge Graph Embedding[J]. computer science, 2015.](https://arxiv.org/pdf/1509.05490.pdf)

[6] [He S ,  Kang L ,  Ji G , et al. Learning to Represent Knowledge Graphs with Gaussian Embedding.  2015.](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf)

