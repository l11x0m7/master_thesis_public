# RACE Reading Comprehension Task

This is a code repo for my master thesis and my conference paper work.

## Work structure

* src: this work consider solving the sentence inference problem.(paper published in CIKM 2018)
* my_src: this work consider different model structures, different priori features and different pre-processing methods, with theano and lasagne framework.
* data: this dir contains the data in the experiments, including original data, preprossed data and embedding data.
* obj: this dir contains models from 'src'
* my_obj: this dir contains models from 'my_src'


## Dependencies

* Python 2
* Theano >= 0.7
* Lasagne == 0.2.dev1

## Datasets

* RACE:
    Please submit a data request [here](http://www.cs.cmu.edu/~glai1/data/race/). The data will be automatically sent to you. Create a "data" directory alongside "src" directory and download the data.

* Word embeddings:
    * glove.6B.zip: [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)

## Usage

### Preprocessing
    * python preprocess.py

### Stanford AR(Lasagne version)
    * test pre-trained model: bash test_SAR.sh
    * train: bash train_SAR.sh (The pre-trained model will be replaced)

### GA(Lasagne version)
    * test pre-trained model: bash test_GA.sh
    * train: bash train_GA.sh (The pre-trained model will be replaced)


## Contact

* Please pull your issues in this repo.

## License

MIT

## References

* [RACE: Large-scale ReAding Comprehension Dataset From Examination](https://arxiv.org/pdf/1704.04683.pdf). Guokun Lai*, Qizhe Xie*, Hanxiao Liu, Yiming Yang and Eduard Hovy. EMNLP 2017
* [code for RACE](https://github.com/qizhex)
