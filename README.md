# BioCreative VII DrugProt Track
This repository contains all material needed for our participation in the BioCreative VII Track 1: Text mining drug and chemical-protein interactions (DrugProt). We explore a Multi-Task Transfer Learning-based method (MTTL) for extracting the chemical-protein relations. We use MTTL by training several clinical and biomedical natural language processing tasks together based on pre-trained language models. The datasets include: DrugProt, ChemProt, TAC 2017, n2c2 2018, bc2gm, bc5cdr, ncbi, i2b2 2010 and ADE dataset. More details are provided [here](https://bionlp.nlm.nih.gov/tac2017adversereactions/).



## Requirements

1. Install all dependencies needed to run this repository:

```
$ pip install -r requirements.txt
```

2. Download the pre-trained model files used in our experiments: 

```
$ bash pretrained_models/download.sh
```


## Quick start

1. Specify the encoder_type in the data_task_def.yml file (SCIBERT or PUBMEDBERT).

2. Load and preprocess the datasets:

```
$ python prepro.py \
  --data_dir "data" \
  --output_dir "data/canonical_data" \
  
$ python prepro_std.py \
  --do_lower_case \
  --root_dir "data/canonical_data" \
  --task_def "data_task_def.yml" \
  --model-type "pubmedbert" \
```

3. Train the model:

```
$ python main.py \
  --init_checkpoint pretrained_models/pubmedbert_base_uncased.pt \
  --task_def data_task_def.yml \
  --train_datasets "chemprot,drugprot,tacrelation,n2c2relation,i2b2relation,ade,bc2gm,bc5cdr,ncbi" \
  --test_datasets "chemprot,drugprot,tacrelation,n2c2relation,i2b2relation,ade,bc2gm,bc5cdr,ncbi" \
```

4. Make predictions:

```
$ python inference.py \
  --guess_file drugprot_test_predictions_mt_pubmedbert_more_data.tsv \
  --gold_dir data/drugprot/test_background \
```



