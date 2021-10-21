#!/usr/bin/env bash


## SCIBERT
cd  "pretrained_models"
mkdir "scibert_vocab"
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/tensorflow_models/scibert_scivocab_uncased.tar.gz -O "scibert_scivocab_uncased.tar.gz"
tar xvf scibert_scivocab_uncased.tar.gz
rm "scibert_scivocab_uncased.tar.gz"
mv "scibert_scivocab_uncased/vocab.txt" "scibert_vocab/vocab.txt"
mv "scibert_scivocab_uncased/bert_model.ckpt.meta" "scibert_vocab/bert_model.ckpt.meta"
mv "scibert_scivocab_uncased/bert_model.ckpt.data-00000-of-00001" "scibert_vocab/bert_model.ckpt.data-00000-of-00001"
mv "scibert_scivocab_uncased/bert_model.ckpt.index" "scibert_vocab/bert_model.ckpt.index"
mv "scibert_scivocab_uncased/bert_config.json" "scibert_vocab/bert_config.json"
rm -r "scibert_scivocab_uncased/"

python convert_tf_to_pt.py \
    --tf-path-model "scibert_vocab/bert_model.ckpt" \
    --config-file "scibert_vocab/bert_config.json" \
    --model-type "scibert" \
    --pytorch-file "scibert_scivocab_uncased.pt" \

rm "scibert_vocab/bert_model.ckpt.meta"
rm "scibert_vocab/bert_model.ckpt.data-00000-of-00001"
rm "scibert_vocab/bert_model.ckpt.index"
rm "scibert_vocab/bert_config.json"

##PUBMEDBERT
cd "pretrained_models"
mkdir "pubmedbert_vocab_a"
wget https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/resolve/main/pytorch_model.bin -O "pytorch_model.bin"
wget https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/resolve/main/vocab.txt -O "pubmedbert_vocab_a/vocab.txt"
wget https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/resolve/main/config.json -O "config.json"

python convert_bin_to_pt.py \
    --bin-model-path "pytorch_model.bin" \
    --config-file "config.json" \
    --pytorch-file "pubmedbert_base_uncased.pt" \

rm "pytorch_model.bin"
rm "config.json"
