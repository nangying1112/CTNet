# Usage

We mainly focus on the [PDTB 2.0](https://www.seas.upenn.edu/~pdtb/) dataset.

Put the folder `PDTB 2.0` to `./` first.

Get all the documents first

Split the document into argument pairs with their corresponding paragraphs as follows:

```
python ./processed_data/split_pairs.py
python ./processed_data/pairs2sentences.py
```

Download [glove.840B.300d.zip](https://nlp.stanford.edu/projects/glove/), unzip and put it to `./`. Further prepare the data:

```
python ./vocab_emb.py
```

Get the adjacent matrix:

```
python ./spacy_coref.py
python ./core_adj.py
python ./adjacency_adj.py
python ./lexical_chain_adj.py
```

Then generate the input of the model:

```
python ./generate_input_data.py
```

For training and evaluating:

```
python ctnet.py \
  --classes 4 \
  --learning_rate 0.001 \
  --batch_size 256 \
  --elmo_cuda 1 \
  --slstm_size 512 \
  --relation_balance True \
  --use_exp True \
  --use_char True \
  --use_mt True 
```


# Requirements

```
python == 3.6
tensorflow == 1.12.0
sklearn == 0.21.3
numpy == 1.17.0
allennlp == 0.9.0
nltk == 3.4.4
spacy == 2.0.12
```

Since tensorflow-hub is unreachable recently, we use ELMo provided by AllenNLP. Download the `option.json` and `weights.hdf5` and put them to `./processed_data/`



