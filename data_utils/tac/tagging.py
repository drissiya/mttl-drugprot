import os
from sys import path
path.append(os.getcwd())
from data_utils.tac.utils import return_overlapping_mentions_per_mention
from data_utils.tac.preprocessing import replace_ponctuation_with_space, process, split_on_uppercase, tokenize_sentence
from nltk.tokenize import word_tokenize

LABELSET = {'AdverseReaction': {'b': 'B-AdverseReaction', 'i': 'I-AdverseReaction', 'l': 'L-AdverseReaction', 'db': 'DB-AdverseReaction', 'di': 'DI-AdverseReaction', 'dl': 'DL-AdverseReaction','u': 'U-AdverseReaction'},
             'Severity': {'b': 'B-Severity', 'i': 'I-Severity', 'l': 'L-Severity', 'db': 'DB-Severity', 'di': 'DI-Severity', 'dl': 'DL-Severity','u': 'U-Severity'},
            'Negation': {'b': 'B-Negation', 'i': 'I-Negation', 'l': 'L-Negation', 'db': 'DB-Negation', 'di': 'DI-Negation', 'dl': 'DL-Negation','u': 'U-Negation'},
            'Factor': {'b': 'B-Factor', 'i': 'I-Factor', 'l': 'L-Factor', 'db': 'DB-Factor', 'di': 'DI-Factor', 'dl': 'DL-Factor','u': 'U-Factor'},
            'Animal': {'b': 'B-Animal', 'i': 'I-Animal', 'l': 'L-Animal', 'db': 'DB-Animal', 'di': 'DI-Animal', 'dl': 'DL-Animal','u': 'U-Animal'},
            'DrugClass': {'b': 'B-DrugClass', 'i': 'I-DrugClass', 'l': 'L-DrugClass', 'db': 'DB-DrugClass', 'di': 'DI-DrugClass', 'dl': 'DL-DrugClass','u': 'U-DrugClass'}}

def tag_mention(tok_text, tok_entity, begin, labelset, labels, entity_start, status, relation_type):
    for i, (word, start) in enumerate(zip(tok_text, entity_start)):
        if start == begin:
            if status == 'simple': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['b'] + '_' + relation_type
                else:
                    labels[i] = labelset['b'] + '_' + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['i'] + '_' + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['i'] + '_' + relation_type
            elif status == 'complex': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['db'] + '_' + relation_type
                else:
                    labels[i] = labelset['db'] + '_' + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['di'] + '_' + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['di'] + '_' + relation_type
            elif status == 'complex_follow': 
                if len(tok_entity) == 1:
                    labels[i] = labelset['di'] + '_' + relation_type
                else:
                    labels[i] = labelset['di'] + '_' + relation_type
                    labels[i+1:i+len(tok_entity)-1] = [labelset['di'] + '_' + relation_type] * (len(tok_entity)-2)
                    labels[i+len(tok_entity)-1] = labelset['di'] + '_' + relation_type                  
    return labels

	
def tagging_sequence_1(set_ADE_mention, sequence_1, tok_text, entity_start, start, end, sentence, section):
    for m in set_ADE_mention:
        start_mention = m[1].split(',')
        len_mention = m[0].split(',')
        mstart = int(start_mention[0])                   
        mend = mstart + int(len_mention[0])  
                        
        if len(start_mention)==1: 
            contin_m, discontin_m = return_overlapping_mentions_per_mention(mstart, mend, set_ADE_mention)
            if len(contin_m)>=1:
                for ment in contin_m:
                    ms = int(ment[1].split(',')[0])                   
                    me = ms + int(ment[0].split(',')[0])  
                    m_str = ment[3]
                    sequence_1 = tagging_sequence(m_str, LABELSET[ment[2]], tok_text, sequence_1, ms, entity_start, 'simple', start, end, len_mention, sentence, ment[5])                                
        else:  
            contin_ms = [] 
            discontin_ms = []
            for s, l in zip(start_mention,len_mention):                          
                contin_m, discontin_m = return_overlapping_mentions_per_mention(int(s),int(s)+int(l), set_ADE_mention)
                contin_ms.extend(contin_m)
                discontin_ms.extend(discontin_m)
                            
            if len(set(discontin_ms))>=1 and len(set(contin_ms))>=1:
                for ment in contin_m:
                    ms = int(ment[1].split(',')[0])                   
                    me = ms + int(ment[0].split(',')[0])  
                    m_str = ment[3]
                    sequence_1 = tagging_sequence(m_str, LABELSET[ment[2]], tok_text, sequence_1, ms, entity_start, 'simple', start, end, len_mention, sentence, ment[5])                                                          
            elif len(set(discontin_ms))>=2 and len(set(contin_ms))==0:
                start_mention_d = discontin_ms[0][1].split(',')
                len_mention_d = discontin_ms[0][0].split(',')
                ms = int(start_mention_d[0])                   
                me = ms + int(len_mention_d[0])  
                sequence_1 = tagging_sequence(section[ms:me], LABELSET[discontin_ms[0][2]], tok_text, sequence_1, ms, entity_start, 'complex', start, end, len_mention, sentence, discontin_ms[0][5])
                for s, l in zip(start_mention_d[1:],len_mention_d[1:]):
                    sequence_1 = tagging_sequence(section[int(s):int(s)+int(l)], LABELSET[discontin_ms[0][2]], tok_text, sequence_1, int(s), entity_start, 'complex_follow', start, end, len_mention, sentence, discontin_ms[0][5])
            elif len(set(discontin_ms))<=1 and len(set(contin_ms))==0:
                sequence_1 = tagging_sequence(section[mstart:mend], LABELSET[m[2]], tok_text, sequence_1, mstart, entity_start, 'complex', start, end, len_mention, sentence, m[5])
                for s, l in zip(start_mention[1:],len_mention[1:]):
                    sequence_1 = tagging_sequence(section[int(s):int(s)+int(l)], LABELSET[m[2]], tok_text, sequence_1, int(s), entity_start, 'complex_follow', start, end, len_mention, sentence, m[5])
    return sequence_1

	
def generate_source_context(ade, tok_text, entity_start, start, end, sentence, section):
    start_mention = ade[1].split(',')
    len_mention = ade[0].split(',')
    mstart = int(start_mention[0])                   
    mend = mstart + int(len_mention[0]) 
    m_str = ade[3] 
    sentence_2 = tokenize_sentence(sentence)
    if len(start_mention)==1: 
        sentence_2 = tagging_sequence(m_str, LABELSET[ade[2]], tok_text, sentence_2, mstart, entity_start, 'simple', start, end, len_mention, sentence, '')                                
    else:  
        sentence_2 = tagging_sequence(section[mstart:mend], LABELSET[ade[2]], tok_text, sentence_2, mstart, entity_start, 'complex', start, end, len_mention, sentence, '')
        for s, l in zip(start_mention[1:],len_mention[1:]):
            sentence_2 = tagging_sequence(section[int(s):int(s)+int(l)], LABELSET[ade[2]], tok_text, sentence_2, int(s), entity_start, 'complex_follow', start, end, len_mention, sentence, '')                        
    return sentence_2
	
def tagging_sequence_2(ade, modifiers, sentence, start, tok_text, end, entity_start, section):
    #Replace ade string with type in the sentence
    sentence_2 = generate_source_context(ade, tok_text, entity_start, start, end, sentence, section)
    #Set sequence 2
    sequence_2 = tagging_sequence_1(modifiers, ['O'] * len(tok_text), tok_text, entity_start, start, end, sentence, section)
    return sentence_2, sequence_2


def tagging_sequence(string, labelset, tok_text, labels, mstart, entity_start, status, start, end, len_mention, ss, relation_type):
    string = process(replace_ponctuation_with_space(string))
    tok_mention = []
    tok = split_on_uppercase(string, True)
    for t in tok:
        tok_mention.extend(word_tokenize(t))
                        
    mention_str_len = len(string.lstrip())
    index = int(len_mention[0]) - mention_str_len

    if mstart in entity_start:                            
        labels = tag_mention(tok_text, tok_mention, mstart, labelset, labels, entity_start, status, relation_type)
    elif (mstart + index) in entity_start:
        labels = tag_mention(tok_text, tok_mention, mstart + index, labelset, labels, entity_start, status, relation_type)
    elif (mstart + index) in range(start, end):
        labels = tag_mention(tok_text, tok_mention, ss.find(tok_mention[0]) + start, labelset, labels, entity_start, status, relation_type)
    return labels

			
			
			
			
