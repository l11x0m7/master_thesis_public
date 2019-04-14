# master_thesis

** Notice: It's been a long time since the experiments were finished, so this repo may not be the same as the original one before(many modifications were done), but you can still look over the methods from this repo. **

Repo for my master degree thesis. This work concludes two main parts:

* Frist, we try to solve or improve the original MRC models with different fusion layers on matching and paraphrasing problems for RACE. Here, we present two heuristic way:
    * Char embedding, NER embedding and co-occurance embedding
    * Hybrid Fusion Layer(stacked with multiple fusion methods)
* Second, we try to improve the sentence inference problem on RACE. Here, we present two module individually:
    * OG: Option Gating
    * KSOG: Key Sentence Option Gating
* Third, experiments on SQuADv1 dataset. You can refer to this repo:[Question_Answering_Models](https://github.com/l11x0m7/Question_Answering_Models). This repo is still updating.
    * BiDAF
    * RNet
    * QANet
    * Hybrid


## Dataset

* RACE:
    Please submit a data request [here](http://www.cs.cmu.edu/~glai1/data/race/). The data will be automatically sent to you. Create a "data" directory alongside "src" directory and download the data.

* Word embeddings:
    * glove.6B.zip: [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## MRC models

* GA Reader(Base)
* SAR(Base)
* BiDAF Model
* Hybrid Model

## Directory Structure

* RACE_for_thesis: 
    This repo mainly contains the work related to my master thesis with the title of "机器阅读理解中的答案排序问题研究".


## Related works

* [An Option Gate Module for Sentence Inference on Machine Reading Comprehension](https://dl.acm.org/citation.cfm?id=3269280). This work has been accepted on CIKM 2018.


