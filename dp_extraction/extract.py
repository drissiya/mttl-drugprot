import os
from sys import path
path.append(os.getcwd())
from nltk.tokenize import word_tokenize
from data_utils.drugprot.brat import Corpus

def extract_mention_from_sentence(pmid, set_toks, ys_bio, start, leng, segment):
    data_pmid = []
    data_toks = []
    data_ys = []
    data_start = []
    data_len = []
    if segment == "BIO":
        for d, tok, ys, st, le in zip(pmid, set_toks, ys_bio, start, leng):
            temp_toks = []
            temp_ys = []
            temp_start = []
            temp_len = []
            temp_pmid = []
            for i, (t, yb, a, b, e) in enumerate(zip(tok, ys, st, le, d)):

                if yb.startswith('B-'):
                    tok_txt = t
                    ys_txt = yb[2:]
                    start_txt = a
                    len_text = b
                    pmid_text = e
                    if (i+1) == len(ys):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_pmid.append(e)
                        break
                    elif ys[i+1].startswith('O') and ys[i-1].startswith('O'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_pmid.append(e)
                    elif ys[i+1].startswith('B-') and ys[i-1].startswith('B-'):
                        temp_toks.append(t)
                        temp_ys.append(yb[2:])
                        temp_start.append(a)
                        temp_len.append(b)
                        temp_pmid.append(e)
                    else: 
                        for k,j in enumerate(ys[i+1:]):
                            if j.startswith('I-'):
                                tok_txt += ' ' + tok[i+k+1]
                                len_text = le[i+k+1] 

                            else:
                                break
                        len_t = len_text
                        temp_toks.append(tok_txt)
                        temp_ys.append(ys_txt)
                        temp_start.append(start_txt)
                        temp_len.append(len_t)
                        temp_pmid.append(pmid_text)
            data_toks.append(temp_toks)
            data_ys.append(temp_ys)
            data_start.append(temp_start)
            data_len.append(temp_len)
            data_pmid.append(temp_pmid)

    return data_pmid, data_toks, data_ys, data_start, data_len

def drugprot_extract_guess_relation(guess, segment, gold_dir):
    dict_modifiers = {}
    dict_relations = {}
    dict_ade = {}


    pmid_relation, toks_relation, type_relation, start_relation, len_relation = extract_mention_from_sentence(guess.t_drug_relation, 
                                                                                                              guess.t_toks_relation, 
                                                                                                              guess.t_segment_relation, 
                                                                                                              guess.t_start_relation, 
                                                                                                              guess.t_len_relation,
                                                                                                              segment)
    
    corpus = Corpus(gold_dir, is_test=True)
    for d in corpus.doc_ids:
        entities_dict = corpus.docs[d].entities_dict1
        text = corpus.docs[d].text
        
        
        relations = extract_modifiers_relation_from_doc(d, text, entities_dict, guess, pmid_relation, toks_relation, type_relation, start_relation, len_relation)
        #dict_modifiers[d] = modifiers
        dict_relations[d] = relations
    return dict_relations

def extract_modifiers_relation_from_doc(m, text, dict_ade, guess, pmid_relation, toks_relation, type_relation, start_relation, len_relation):
    relations = []
    modifiers = []

    for ade, drug_r, toks_r, type_r, start_r, len_r in zip(guess.t_ade, pmid_relation, toks_relation,type_relation, start_relation, len_relation):
        if len(set(drug_r))==0:
            continue
        if list(set(drug_r))[0]==m:
            arg1 = ade[4]

            for tok, typ, st, le in zip(toks_r, type_r, start_r, len_r): 
                relation_type = typ.split('_')[1]

                modifier = (int(st), int(le), typ.split('_')[0], text[int(st):int(le)])
                if modifier in dict_ade.keys():
                    
                    arg2 = dict_ade[modifier]    
                    #modifiers.append(modifier)

                    relations.append((m, arg1, arg2, relation_type))


    #print ("==============")
    return relations