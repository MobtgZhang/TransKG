# TransKG:The translation model for knowledge Graph
Knowledge Graph Embedding model collections implemented by Pytorch.Including TransE<sup>1</sup>,TransH<sup>2</sup>,TransR<sup>3</sup>,TransD<sup>4</sup>,TransA<sup>5</sup>,KG2E<sup>6</sup> models for Knowledge Graph.
# Requirements
+ tqdm>=4.59.0
+ torch>=1.8.0
+ pandas>=1.2.3
+ numpy>=1.20.1
+ tensorboardX>=2.2
# Usage
Step1: download the FB15K-237 dataset,then extract the dataset file into the file `data`. 

Step2: Use following command to train the embedding.
```bash
python main.py
```
Then the embedding file will save in dir:`./checkpoints`
# Dataset
The FB15K-237 dataset is here:[FB15K-237](https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz).
# References
[1] [Bordes A ,  Usunier N ,  Garcia-Duran A , et al. Translating Embeddings for Modeling Multi-relational Data. Curran Associates Inc.  2013.](http://www.thespermwhale.com/jaseweston/papers/CR_paper_nips13.pdf)

[2] [Zhang J . Knowledge Graph Embedding by Translating on Hyperplanes[J]. AAAI - Association for the Advancement of Artificial Intelligence, 2015.](http://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf)

[3] [Learning Entity and Relation Embeddings for Knowledge Graph Completion Lin Y, Liu Z, Zhu X, et al. AAAI. 2015. ](http://nlp.csai.tsinghua.edu.cn/~lyk/publications/aaai2015_transr.pdf)

[4] [Ji G ,  He S ,  Xu L , et al. Knowledge Graph Embedding via Dynamic Mapping Matrix[C]// Meeting of the Association for Computational Linguistics & the International Joint Conference on Natural Language Processing. 2015.](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Knowledge%20Graph%20Embedding%20via%20Dynamic%20Mapping%20Matrix.pdf)

[5] [Xiao H ,  Huang M ,  Hao Y , et al. TransA: An Adaptive Approach for Knowledge Graph Embedding[J]. computer science, 2015.](https://arxiv.org/pdf/1509.05490.pdf)

[6] [He S ,  Kang L ,  Ji G , et al. Learning to Represent Knowledge Graphs with Gaussian Embedding.  2015.](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Learning%20to%20Represent%20Knowledge%20Graphs%20with%20Gaussian%20Embedding.pdf)

