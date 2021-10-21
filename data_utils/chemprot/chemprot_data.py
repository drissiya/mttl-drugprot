import os
import spacy
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from xml.etree import ElementTree
from sklearn.model_selection import ShuffleSplit
from sys import path
path.append(os.getcwd())
from data_utils.chemprot.brat import Corpus
from data_utils.chemprot.utils import get_concepts, get_relations_from_sentence
from data_utils.chemprot.preprocessing import replace_ponctuation_with_space, spans, tokenize_sentence, process
from data_utils.chemprot.tagging import tagging_sequence_2

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = nlp.tokenizer.tokens_from_list

def load_data(dir_name, segment, test):
    chem_relation_data = []
    corpus = Corpus(dir_name, is_test=test)
    for d in corpus.doc_ids:
        entities = corpus.docs[d].entities
        relations = corpus.docs[d].relations
        text = corpus.docs[d].text

        punkt_tokenizer = PunktSentenceTokenizer()
        punkt_tokenizer._params.abbrev_types.add('dr')

        for line in text.split('\n'):
            sent_spans = list(punkt_tokenizer.span_tokenize(line))
            sent_texts = list(punkt_tokenizer.sentences_from_text(line))
            for sent_span, sent_text in zip(sent_spans, sent_texts):

                start_sent = sent_span[0]
                end_sent = sent_span[1]
                genes, chemical, concept_sent = get_concepts(entities, start_sent, end_sent)
                if len(genes)> 0 and len(chemical)>0:
					
                    sentence = process(replace_ponctuation_with_space(sent_text))
                    tok_text = word_tokenize(sentence)
                    entity_start, entity_end = spans(sent_text, tok_text, start_sent)
                    entity_file = [d]*len(tok_text)
                    chem_relation = get_relations_from_sentence(chemical, genes, relations)


                    for c, att in chem_relation:
                        c1 = (str(c.end), str(c.start), c.ttype, c.text, c.tid)
                        att1 = []
                        for a in att:
                            att1.append((str(a[0].end), str(a[0].start), a[0].ttype, a[0].text))
                        #print (att1)
                        #print (c)
                        sentence_2, sequence_2 = tagging_sequence_2(d, 
                                                                    c, 
                                                                    att, 
                                                                    sentence, 
                                                                    start_sent, 
                                                                    tok_text, 
                                                                    end_sent, 
                                                                    entity_start, 
                                                                    segment)
                        chem_relation_data.append((sentence_2,tok_text, sequence_2,entity_start,entity_file,entity_end,c1, att1)) 
    return chem_relation_data
	
	
class ChemProt:
    def __init__(self, segment = "BIO", test=False, data_dir=None):
        self.data_dir = data_dir
        self.segment = segment
        self.test = test

        
        self.ade_relation_data = []
        self.t_section_relation = []
        self.t_toks_relation = []
        self.t_sentence_relation = []
        self.t_segment_relation = []
        self.t_start_relation = []
        self.t_len_relation = []
        self.t_ade = []
        self.t_modifiers = []  
        self.t_drug_relation = []  
        
        #self.load_corpus()

    def load_corpus(self):

        ADE_relation_data = load_data(self.data_dir, self.segment, self.test)
        self.ade_relation_data = ADE_relation_data        
        for seq in self.ade_relation_data:
            self.t_sentence_relation.append(seq[0])
            self.t_toks_relation.append(seq[1])
            self.t_segment_relation.append(seq[2])
            self.t_start_relation.append(seq[3])
            self.t_section_relation.append(seq[4])
            self.t_len_relation.append(seq[5])
            self.t_ade.append(seq[6])
            self.t_modifiers.append(seq[7])
            self.t_drug_relation.append(seq[4])                



