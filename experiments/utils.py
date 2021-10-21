import os
from sys import path
path.append(os.getcwd())
from data_utils.task_def import DataFormat
from experiments.i2b2_utils import *

def load_adr_ner(tac_data):
    rows = []
    cnt = 0
    for sentence, label, start, section, lenn, drug, sentence_input, start_sentence, len_sentence in zip(tac_data.t_toks_mention, tac_data.t_segment_mention, tac_data.t_start_mention, tac_data.t_section_mention, tac_data.t_len_mention, tac_data.t_drug_mention, tac_data.t_sentence_input_mention, tac_data.t_start_sentence_mention, tac_data.t_len_sentence_mention):
        sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': start, 'section': section, 'lenn': lenn, 'drug': drug, 'sentence_input': [sentence_input], 'start_sentence': [start_sentence], 'len_sentence': [len_sentence], 'token': sentence, 'ade': sentence, 'modifier': sentence}
        rows.append(sample)
        cnt += 1
    return rows
	
def load_adr_relation(tac_data):
    rows = []
    cnt = 0
    for sentence, label, ade, token, start, section, lenn, drug, modifier in zip(tac_data.t_sentence_relation, tac_data.t_segment_relation, tac_data.t_ade, tac_data.t_toks_relation, tac_data.t_start_relation, tac_data.t_section_relation, tac_data.t_len_relation, tac_data.t_drug_relation, tac_data.t_modifiers):
        sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': start, 'section': section, 'lenn': lenn, 'drug': drug, 'sentence_input': token, 'start_sentence': token, 'len_sentence': token, 'token': token, 'ade': list(ade), 'modifier': list(modifier)}
        rows.append(sample)
        cnt += 1 
    return rows
	
def load_ner(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label= []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split('\t')
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
    return rows
	
	
def load_i2b2_2010_concept(beth_path, partners_path):
    rows = []
    cnt = 0
    all_tokens_beth, all_concepts_beth = parse_dir(beth_path)
    all_tokens_partners, all_concepts_partners = parse_dir(partners_path)
    all_tokens = all_tokens_beth + all_tokens_partners
    all_concepts = all_concepts_beth + all_concepts_partners
    for sentence, label in zip(all_tokens,all_concepts):
        sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
        rows.append(sample)
        cnt += 1
    return rows
	
def load_i2b2_2010_concept_test(test_path):
    rows = []
    cnt = 0
    all_tokens, all_concepts = parse_dir(test_path)
    for sentence, label in zip(all_tokens,all_concepts):
        sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
        rows.append(sample)
        cnt += 1
    return rows
	
def load_i2b2_2010_relation(beth_path, partners_path):
    rows = []
    cnt = 0
    beth_con_path = beth_path + 'concept/'
    beth_rel_path = beth_path + 'rel/'
    beth_txt_path = beth_path + 'txt/'
    data_beth = get_dataset_dataframe_classification(beth_con_path, beth_rel_path, beth_txt_path)
    
    partners_con_path = partners_path + 'concept/'
    partners_rel_path = partners_path + 'rel/'
    partners_txt_path = partners_path + 'txt/'
    data_partners = get_dataset_dataframe_classification(partners_con_path, partners_rel_path, partners_txt_path)
    
    data = data_beth + data_partners
    for seq in data:
        sample = {'uid': cnt, 'premise': seq[0], 'label': seq[1]}
        rows.append(sample)
        cnt += 1
    return rows
	
def load_i2b2_2010_relation_test(test_path):
    rows = []
    cnt = 0
    beth_con_path = test_path + 'concept/'
    beth_rel_path = test_path + 'rel/'
    beth_txt_path = test_path + 'txt/'
    data = get_dataset_dataframe_classification(beth_con_path, beth_rel_path, beth_txt_path)

    for seq in data:
        sample = {'uid': cnt, 'premise': seq[0], 'label': seq[1]}
        rows.append(sample)
        cnt += 1
    return rows
	
def load_ade_corpus(X_data, y_data):
    rows = []
    cnt = 0
    for sentence, label in zip(X_data, y_data):
        sample = {'uid': cnt, 'premise': sentence, 'label': label}
        rows.append(sample)
        cnt += 1
    return rows
    
	
def load_conll_ner(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label= []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[-1])
        if len(sentence) > 0:
            sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
    return rows

def load_conll_pos(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label= []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[1])
        if len(sentence) > 0:
            sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
    return rows

def load_conll_chunk(file, is_train=True):
    rows = []
    cnt = 0
    sentence = []
    label= []
    with open(file, encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
                if len(sentence) > 0:
                    sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
                    rows.append(sample)
                    sentence = []
                    label = []
                    cnt += 1
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            label.append(splits[2])
        if len(sentence) > 0:
            sample = {'uid': cnt, 'premise': sentence, 'label': label, 'start': sentence, 'section': sentence, 'lenn': sentence, 'drug': sentence, 'sentence_input': sentence, 'start_sentence': sentence, 'len_sentence': sentence, 'token': sentence, 'ade': sentence, 'modifier': sentence}
    return rows
