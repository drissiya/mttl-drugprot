import os
from sys import path
path.append(os.getcwd())
from data_utils.drugprot.preprocessing import replace_ponctuation_with_space, tokenize_sentence, process
from nltk.tokenize import word_tokenize


LABELSET = {'CHEMICAL': {'b': 'B-CHEMICAL', 'i': 'I-CHEMICAL', 'l': 'L-CHEMICAL', 'u': 'U-CHEMICAL'},
             'GENE-Y': {'b': 'B-GENE', 'i': 'I-GENE', 'l': 'L-GENE', 'u': 'U-GENE'},
            'GENE-N': {'b': 'B-GENE', 'i': 'I-GENE', 'l': 'L-GENE', 'u': 'U-GENE'},
			'GENE': {'b': 'B-GENE', 'i': 'I-GENE', 'l': 'L-GENE', 'u': 'U-GENE'}}

def tag_sentence(tok_text, tok_entity, begin, labelset, labels, entity_start, relation_type, segment):
    if segment == "BIO":
        for i, (word, start) in enumerate(zip(tok_text, entity_start)):
            if start == begin: 
                if len(tok_entity) == 1:
                    labels[i] = labelset['b'] + relation_type
                else:
                    labels[i] = labelset['b'] + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i'] + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['i'] + relation_type
    elif segment == "BILOU":
        for i, (word, start) in enumerate(zip(tok_text, entity_start)):
            if start == begin: 
                if len(tok_entity) == 1:
                    labels[i] = labelset['u'] + relation_type
                else:
                    labels[i] = labelset['b'] + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i'] + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['l'] + relation_type
    return labels

def return_label(lab, string, labelset, tok_text, labels, mstart, entity_start, start, end, ss, relation_type, segment):
    string = process(replace_ponctuation_with_space(string))
    tok_mention = word_tokenize(string)
                                                  
    labels = tag_sentence(tok_text, tok_mention, mstart, labelset, labels, entity_start, relation_type, segment)
    return labels

def tagging_sequence_1(lab, set_ADE_mention, sequence_1, tok_text, entity_start, start, end, sentence, segment):
    for m in set_ADE_mention:
        mstart = m[0].start                   
        mend = m[0].end    
        m_str = m[0].text

        sequence_1 = return_label(lab, m_str, LABELSET[m[0].ttype], tok_text, sequence_1, mstart, entity_start, start, end, sentence, '_' + m[1], segment)                                
    return sequence_1

def replace_tokens_with_ADE_type(lab, drug, tok_text, entity_start, start, end, sentence, segment):
    mstart = drug.start                  
    mend = drug.end 
    m_str = drug.text
    sentence_2 = word_tokenize(sentence) 
    sentence_2 = return_label(lab, m_str, LABELSET[drug.ttype], tok_text, sentence_2, mstart, entity_start, start, end, sentence, '', segment)                                
    return sentence_2

def tagging_sequence_2(lab, drug, attributes, sentence, start, tok_text, end, entity_start, segment):
    #Replace ade string with type in the sentence
    sentence_2 = replace_tokens_with_ADE_type(lab, drug, tok_text, entity_start, start, end, sentence, segment)
    #Set sequence 2
    
    sequence_2 = tagging_sequence_1(lab, attributes, ['O'] * len(tok_text), tok_text, entity_start, start, end, sentence, segment)
    return sentence_2, sequence_2


			
			
			
			
