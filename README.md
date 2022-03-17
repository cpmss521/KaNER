# KaNER


###  Examples Instructions
(1) Train NCBI on train dataset, evaluate on dev dataset:
```
python ./spert.py train --config configs/example_train.conf
```

(2) Evaluate the NCBI model on test dataset:
```
python ./spert.py eval --config configs/example_eval.conf
```

### Fetch data
datasets lies in data file

### Additional files:
+ The files `Knowledge_Vocabulary.json`, `Concept description information.txt` can be extracted directly from UMLS by  MetaMap tool(to use UMLS, you need to request [access permission](https://www.nlm.nih.gov/research/umls/index.html)). Note that some UMLS concepts may not have any definition sentence.
+ `Knowledge_Embeddings.npy` contains the UMLS embeddings of [Maldonado et al. 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6568073/).
+ scispaCy  preprocess the concept description information 


