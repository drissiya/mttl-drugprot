import os
import spacy
from nltk.tokenize import word_tokenize
from xml.etree import ElementTree
from sklearn.model_selection import ShuffleSplit
from sys import path
path.append(os.getcwd())
from data_utils.tac.drug_label import xml_files, read
from data_utils.tac.preprocessing import process, replace_ponctuation_with_space, spans, tokenize_sentence
from data_utils.tac.utils import get_section, get_mentions, get_relations, get_mentions_relations_from_sentence
from data_utils.tac.tagging import tagging_sequence_1, tagging_sequence_2


def load_dir(dir_name_labels, dir_name_sentence):
    """
    Args:
        dir_name_labels: directory that contains all xml files with Section, Mention and Relation elements.
        dir_name_sentence: directory that contains all xml files with Section and Sentence elements.
    """
    #validate_ind function return the sections and mentions of a given label written in XML format (e.g. 'ACTEMRA')
    ADE_rel_mention_data = []
    ADE_no_rel_mention_data = []
    ADE_relation_data = []
    files = xml_files(dir_name_labels)
    for key, value in zip(files.keys(), files.values()):
        label = read(value)
        sections = label.sections
        mentions = label.mentions
        relations = label.relations

        root = ElementTree.parse(os.path.join(dir_name_sentence, key + '.xml')).getroot()
        assert root.tag == 'Label', 'Root is not Label: ' + root.tag
        assert root[0].tag == 'Section', 'Expected \'Text\': ' + root[0].tag
    
        for sec in root: 
            #Return id_section and section
            id_section, section = get_section(sections, sec)
            
            #Return mentions for a given id_section            
            unique_section_mentions, section_mentions = get_mentions(mentions, id_section)
            
            #Return relations with complete argument
            unique_relations = get_relations(section_mentions, relations)
            
            #Return sentence from the offset.
            for sent in sec:
                start = int(sent.attrib['start'])
                end = int(sent.attrib['len'])
                
                #Ignore empty sentence
                sentence = section[start:end]
                if len(process(replace_ponctuation_with_space(sentence)).strip()) == 0:
                    continue
                
                #Tokenize sentence
                sentence = process(replace_ponctuation_with_space(sentence))
                tok_text = tokenize_sentence(sentence)
                
                #Map each token to section, start and len
                entity_section = [id_section]*len(tok_text)
                entity_drug = [key]*len(tok_text)
                entity_start, entity_end = spans(sentence, tok_text, start)
                 
                set_ADE_mention, set_ADE_relation = get_mentions_relations_from_sentence(unique_section_mentions, start, end, unique_relations)

                
                #Initialise and set sequence_1 for source mentions
                sequence_1 = ['O'] * len(tok_text) 
				
                if len(set_ADE_mention)>0:
                    sequence_1 = tagging_sequence_1(set_ADE_mention, sequence_1, tok_text, entity_start, start, end, sentence, section)
                    ADE_rel_mention_data.append((tok_text,sequence_1,entity_start,entity_section,entity_end, entity_drug, sentence, start, end, section))

                if len(set_ADE_relation)>0:                     
                    for ade, modifiers in set_ADE_relation:
                        sentence_2, sequence_2 = tagging_sequence_2(ade, modifiers, sentence, start, tok_text, end, entity_start, section)
                        ADE_relation_data.append((sentence_2,tok_text, sequence_2,entity_start,entity_section,entity_end,ade, modifiers, entity_drug))						
    return ADE_rel_mention_data, ADE_relation_data

def split_data_sequence_tac(sentences_train):
    TAC_dev = TAC()
    TAC_train_temp = TAC()
    
    rs = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    for train_index, test_index in rs.split(sentences_train.t_toks_mention):
        TAC_dev.t_toks_mention = [sentences_train.t_toks_mention[i] for i in test_index]
        TAC_dev.t_segment_mention = [sentences_train.t_segment_mention[i] for i in test_index]
        TAC_dev.t_start_mention = [sentences_train.t_start_mention[i] for i in test_index]
        TAC_dev.t_section_mention = [sentences_train.t_section_mention[i] for i in test_index]
        TAC_dev.t_len_mention = [sentences_train.t_len_mention[i] for i in test_index]
        TAC_dev.t_drug_mention = [sentences_train.t_drug_mention[i] for i in test_index]
        TAC_dev.t_sentence_input_mention = [sentences_train.t_sentence_input_mention[i] for i in test_index]
        TAC_dev.t_start_sentence_mention = [sentences_train.t_start_sentence_mention[i] for i in test_index]
        TAC_dev.t_len_sentence_mention = [sentences_train.t_len_sentence_mention[i] for i in test_index]
        
        TAC_train_temp.t_toks_mention = [sentences_train.t_toks_mention[i] for i in train_index]
        TAC_train_temp.t_segment_mention = [sentences_train.t_segment_mention[i] for i in train_index]
        TAC_train_temp.t_start_mention = [sentences_train.t_start_mention[i] for i in train_index]
        TAC_train_temp.t_section_mention = [sentences_train.t_section_mention[i] for i in train_index]
        TAC_train_temp.t_len_mention = [sentences_train.t_len_mention[i] for i in train_index]
        TAC_train_temp.t_drug_mention = [sentences_train.t_drug_mention[i] for i in train_index]
        TAC_train_temp.t_sentence_input_mention = [sentences_train.t_sentence_input_mention[i] for i in train_index]
        TAC_train_temp.t_start_sentence_mention = [sentences_train.t_start_sentence_mention[i] for i in train_index]
        TAC_train_temp.t_len_sentence_mention = [sentences_train.t_len_sentence_mention[i] for i in train_index]

    for train_index, test_index in rs.split(sentences_train.t_sentence_relation):
        TAC_dev.t_sentence_relation = [sentences_train.t_sentence_relation[i] for i in test_index]
        TAC_dev.t_segment_relation = [sentences_train.t_segment_relation[i] for i in test_index]
        TAC_dev.t_toks_relation = [sentences_train.t_toks_relation[i] for i in test_index]
        TAC_dev.t_start_relation = [sentences_train.t_start_relation[i] for i in test_index]
        TAC_dev.t_section_relation = [sentences_train.t_section_relation[i] for i in test_index]
        TAC_dev.t_len_relation = [sentences_train.t_len_relation[i] for i in test_index]
        TAC_dev.t_ade = [sentences_train.t_ade[i] for i in test_index]
        TAC_dev.t_modifiers = [sentences_train.t_modifiers[i] for i in test_index]
        TAC_dev.t_drug_relation = [sentences_train.t_drug_relation[i] for i in test_index]
        
        TAC_train_temp.t_sentence_relation = [sentences_train.t_sentence_relation[i] for i in train_index]
        TAC_train_temp.t_segment_relation = [sentences_train.t_segment_relation[i] for i in train_index]
        TAC_train_temp.t_toks_relation = [sentences_train.t_toks_relation[i] for i in train_index]
        TAC_train_temp.t_start_relation = [sentences_train.t_start_relation[i] for i in train_index]
        TAC_train_temp.t_section_relation = [sentences_train.t_section_relation[i] for i in train_index]
        TAC_train_temp.t_len_relation = [sentences_train.t_len_relation[i] for i in train_index]
        TAC_train_temp.t_ade = [sentences_train.t_ade[i] for i in train_index]
        TAC_train_temp.t_modifiers = [sentences_train.t_modifiers[i] for i in train_index]
        TAC_train_temp.t_drug_relation = [sentences_train.t_drug_relation[i] for i in train_index]

    return TAC_train_temp, TAC_dev 

	
class TAC:
    def __init__(self, sentence_dir=None, label_dir=None):
        self.sentence_dir = sentence_dir
        self.label_dir = label_dir
               
        self.ade_source_data = []
        self.t_drug_mention = []
        self.t_toks_mention = []
        self.t_segment_mention = []
        self.t_section_mention = []
        self.t_start_mention = []
        self.t_len_mention = []
        self.t_sentence_input_mention = []
        self.t_start_sentence_mention = []
        self.t_len_sentence_mention = []
        self.t_section_text_mention = []
        
        self.ade_att_rel_data = []
        self.t_drug_relation = []
        self.t_toks_relation = []
        self.t_sentence_relation = []
        self.t_segment_relation = []
        self.t_section_relation = []
        self.t_start_relation = []
        self.t_len_relation = []
        self.t_ade = []
        self.t_modifiers = []       


    def load_corpus(self):
        ade_source_data, ade_att_rel_data = load_dir(self.label_dir, self.sentence_dir)
        self.ade_source_data = ade_source_data
        self.ade_att_rel_data = ade_att_rel_data
        
        for seq in self.ade_att_rel_data:
            self.t_sentence_relation.append(seq[0])
            self.t_toks_relation.append(seq[1])
            self.t_segment_relation.append(seq[2])
            self.t_start_relation.append(seq[3])
            self.t_section_relation.append(seq[4])
            self.t_len_relation.append(seq[5])
            self.t_ade.append(seq[6])
            self.t_modifiers.append(seq[7])
            self.t_drug_relation.append(seq[8])

            
            
        for seq in self.ade_source_data:
            self.t_toks_mention.append(seq[0])
            self.t_segment_mention.append(seq[1])
            self.t_start_mention.append(seq[2])
            self.t_section_mention.append(seq[3])
            self.t_len_mention.append(seq[4])
            self.t_drug_mention.append(seq[5])
            self.t_sentence_input_mention.append(seq[6])
            self.t_start_sentence_mention.append(seq[7])
            self.t_len_sentence_mention.append(seq[8])
            self.t_section_text_mention.append(seq[9])