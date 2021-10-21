
# coding: utf-8

# In[3]:


import os
import re
import tarfile
import shutil
import subprocess
import pickle
import glob
import os
import re
import pandas as pd
from tqdm import tqdm

relation_dict = {0: 'TrIP', 1: 'TrWP', 2: 'TrCP', 3: 'TrAP', 4: 'TrNAP', 5: 'TeRP', 6: 'TeCP', 7: 'PIP', 8:'None'}
#relation_dict = {0: 'TrIP', 1: 'TrWP', 2: 'TrCP', 3: 'TrAP', 4: 'TrNAP', 5: 'TeRP', 6: 'TeCP', 7: 'PIP', 8: 'None'}
rev_relation_dict = {'TrIP': 0, 'TrWP': 1, 'TrCP': 2, 'TrAP': 3, 'TrNAP': 4, 'TeRP': 5, 'TeCP': 6, 'PIP': 7, 
        'TrP-None': 8, 'TeP-None': 8, 'PP-None': 8}

def parse_dir(base_path):
    base_txt_path = base_path + 'txt/'
    base_con_path = base_path + 'concept/'

    all_txt_files = os.listdir(base_txt_path)
    all_txt_files = [item for item in all_txt_files if item[-3:] == 'txt']
    all_txt_files.sort()

    all_tokens = []
    all_concepts = []

    for txt_filename in all_txt_files:
        # read text file
        text = open(base_txt_path + txt_filename, 'r', encoding='utf-8-sig').read()
        token_list = [re.split('\ +', sentence) for sentence in text.split('\n')]
        token_list = [sentence for sentence in token_list if len(sentence) > 0]

        # read concept file
        concepts = open(base_con_path + txt_filename[:-3] + 'con', 'r', encoding='utf-8-sig').read()
        concepts = concepts.split('\n')
        concepts = [concept_item for concept_item in concepts if len(concept_item) > 1]

        # build annotation
        concepts_list = [['O'] * len(sentence) for sentence in token_list]

        for concept_item in concepts:
            concept_name = re.findall(r'c="(.*?)" \d', concept_item)[0]
            concept_tag = re.findall(r't="(.*?)"$', concept_item)[0]

            concept_span_string = re.findall(r'(\d+:\d+\ \d+:\d+)', concept_item)[0]

            span_1, span_2 = concept_span_string.split(' ')
            line1, start = span_1.split(':')
            line2, end = span_2.split(':')

            assert line1 == line2

            line1, start, end = int(line1), int(start), int(end)

            concept_name = re.sub(r'\ +', ' ', concept_name)
            original_text = ' '.join(token_list[line1 - 1][start:end + 1])

            if concept_name != original_text.lower():
                print(concept_name, original_text)
                raise RuntimeError

            first = True
            for start_id in range(start, end + 1):
                if first:
                    concepts_list[line1 - 1][start_id] = 'B-' + concept_tag
                    first = False
                else:
                    concepts_list[line1 - 1][start_id] = 'I-' + concept_tag

        all_tokens += (token_list)
        all_concepts += (concepts_list)

    return all_tokens, all_concepts



# given a file path, just get the name of the file
def get_filename_with_extension(path):
    return os.path.basename(path)

# given the file name with an extension like filename.con, return the filename 
# without the extension i.e. filename
def get_filename_without_extension(path):
    filename_with_extension = os.path.basename(path)
    return os.path.splitext(filename_with_extension)[0]

# given a string that looks like c="concept" extract the concept
def extract_concept_from_string(fullstring):
    return re.match(r'^c=\"(?P<concept>.*)\"$', fullstring).group('concept')

# given a string that looks like t="type" extract the type
def extract_concept_type_from_string(fullstring):
    return re.match(r'^t=\"(?P<type>.*)\"$', fullstring).group('type')

# given a string that looks like r="TrAP" extract the relation
def extract_relation_from_string(fullstring):
    return re.match(r'^r=\"(?P<relation>.*)\"$', fullstring).group('relation')

# given a concept that looks like c="his home regimen" 111:8 111:10, return the components
def get_concept_subparts(concept):
    concept_name = " ".join(concept.split(' ')[:-2])
    concept_name = extract_concept_from_string(concept_name)

    concept_pos1 = concept.split(' ')[-2]
    concept_pos2 = concept.split(' ')[-1]
    return concept_name, concept_pos1, concept_pos2

# given a position like 111:8 return the line number and word number
def get_line_number_and_word_number(position):
    split = position.split(':')
    return split[0], split[1]

# given a specific concept file path, generate a concept dictionary
def get_concept_dictionary(file_path):
    concept_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            concept = line.split('||')[0] # line splitting
            type_of_concept = line.split('||')[1]
            
            type_of_concept = extract_concept_type_from_string(type_of_concept) # getting useful info
            concept_name, concept_pos1, concept_pos2 = get_concept_subparts(concept)

            line1, _ = get_line_number_and_word_number(concept_pos1)
            line2, _ = get_line_number_and_word_number(concept_pos2)
            if line1 != line2:
                print("There is a problem! Concept spans multiple lines")
            
            from_to_positions = concept_pos1 + ";" + concept_pos2
            concept_dict[from_to_positions] = {
                    'fromto': from_to_positions, 'word': concept_name, 'type': type_of_concept}
    return concept_dict

# given a line number and the concept dictionary, return all the concepts from the 
# particular line #
def get_entity_replacement_dictionary(linenum, concept_dict):
    entity_replacement = {}
    for key, val in concept_dict.items():
        dict_linenum = key.split(';')[0].split(':')[0]
        if dict_linenum == linenum:
            fromword = key.split(';')[0].split(':')[1]
            toword = key.split(';')[1].split(':')[1]
            ent_repl_key = str(fromword) + ':' + str(toword)
            entity_replacement[ent_repl_key] = val['type']
    return entity_replacement # returns a list of dictionaries i.e. from-to, word, type

# given a line in the relation file, return the concept1 word, spans, relation and concept 2 word, spans
def read_rel_line(rel_line):
    line = rel_line.strip()
    concept1 = line.split('||')[0]
    relation = line.split('||')[1]
    concept2 = line.split('||')[2]

    concept1_name, concept1_pos1, concept1_pos2 = get_concept_subparts(concept1)
    concept2_name, concept2_pos1, concept2_pos2 = get_concept_subparts(concept2)
    relation = extract_relation_from_string(relation)

    line1_concept1, from_word_concept1 = get_line_number_and_word_number(concept1_pos1)
    line2_concept1, to_word_concept1  = get_line_number_and_word_number(concept1_pos2)

    line1_concept2, from_word_concept2 = get_line_number_and_word_number(concept2_pos1)
    line2_concept2, to_word_concept2 = get_line_number_and_word_number(concept2_pos2)

    if line1_concept1 != line2_concept1 or line1_concept2 != line2_concept2 or             line1_concept1 != line1_concept2:
                print("Concepts are in two different lines")
    # assuming that all the lines are the same
    return {'e1_word': concept1_name, 'e1_from': from_word_concept1, 'e1_to': to_word_concept1,
            'e2_word': concept2_name, 'e2_from': from_word_concept2, 'e2_to': to_word_concept2, 
            'line_num': line1_concept1, 'relation': relation}

# below is for the case that you do not want to extract the None relations from the data, 
# because that is inferred from the concept types rather than explicitly present in the 
# relation annotations
# give it a directory with res(directory + 'concept/')
def get_dataset_dataframe_classification(concept_directory, rel_directory, txt_directory):
    data = []
    total_rel_files_to_read = glob.glob(os.path.join(rel_directory, '*'))
    
    for rel_file_path in total_rel_files_to_read:
        with open(rel_file_path, 'r') as rel_file:
            base_filename = get_filename_without_extension(rel_file_path)
            concept_file_path = os.path.join(concept_directory, base_filename +".con")
            concept_dictionary = get_concept_dictionary(concept_file_path)
            
            text_file_path = os.path.join(txt_directory, base_filename +".txt")
            text_file = open(text_file_path, 'r').readlines() 

            for rel_line in rel_file:
                rel_dict = read_rel_line(rel_line)
                tokenized_sentence = text_file[int(rel_dict['line_num']) - 1].strip()
                sentence_text = tokenized_sentence
                tokens = sentence_text.split(' ')
                e1 = rel_dict['e1_word']
                e2 = rel_dict['e2_word']
                relation_type = rel_dict['relation']
                linenum = rel_dict['line_num']
                entity_replacement_dict = get_entity_replacement_dictionary(linenum, concept_dictionary)

                e1_idx = (int(rel_dict['e1_from']), int(rel_dict['e1_to']))
                e2_idx = (int(rel_dict['e2_from']), int(rel_dict['e2_to']))
                if e1_idx[0] < e2_idx[0]:
                    context = tokens[:e1_idx[0]] + ["CONCEPT 1"] + tokens[e1_idx[1]+1:e2_idx[0]] + ["CONCEPT 2"] + tokens[e2_idx[1]+1:]
                else:
                    context = tokens[:e2_idx[0]] + ["CONCEPT 2"] + tokens[e2_idx[1]+1:e1_idx[0]] + ["CONCEPT 1"] + tokens[e1_idx[1]+1:]
                data.append([context, str(relation_type)])

    return data
