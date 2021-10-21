import os
import codecs
from warnings import warn
from collections import defaultdict

ENCODING = 'utf-8'

class Entity():

    def __init__(self, splits):
        self.pmid = splits[0]
        self.tid = splits[1]
        self.ttype = splits[2]
        self.start = int(splits[3])
        self.end = int(splits[4])
        self.text = splits[5]
        
    def __str__(self):
        """String representation."""
        return '{}\t{}\t{}\t({}:{})'.format(self.pmid, self.ttype, self.text, self.start, self.end)


class Relation():
    type = 'na'

    def __init__(self, pmid, arg1, arg2, rtype):
        """Init."""
        assert isinstance(arg1, Entity)
        assert isinstance(arg2, Entity)
        self.pmid = str(pmid).strip()
        self.arg1 = arg1
        self.arg2 = arg2
        self.rtype = str(rtype).strip()


    def __str__(self):
        """String representation."""
        return '{} ({}->{})'.format(self.rtype, self.arg1.ttype,
                                    self.arg2.ttype)
    
class Document(object):
    def __init__(self, doc_id, text):

        self.doc_id = doc_id

        self.no_text = True
        self.text = text

        self.annot = []
        self.entities = []
        self.entities_dict = dict()
        self.entities_dict1 = dict()
        self.relations = []     
        #self.relations_dict = defaultdict(dict)     


class Corpus(object):
    

    def __init__(self, corpus_dir, is_test=False):
        self.basename = os.path.basename(corpus_dir)
        self.txt_path = os.path.join(corpus_dir, self.basename + '_abstracts.tsv')
        self.ent_path = os.path.join(corpus_dir, self.basename + '_entities.tsv')
        self.is_test = is_test
        if not is_test:
            self.rel_path = os.path.join(corpus_dir, self.basename + '_gold_standard.tsv')

        self.docs = dict()
        self.doc_ids = set()

        self.split_doc()


    def split_doc(self):
        with codecs.open(self.txt_path, encoding=ENCODING) as f_txt:
            for line in f_txt:
                (pmid, title, text) = line.strip().split('\t')
                self.doc_ids.add(pmid) 
                self.docs[pmid] = Document(pmid, title + '\t' + text)

        print ("# of docs: %d" % len(self.docs))

        with codecs.open(self.ent_path, encoding=ENCODING) as f_ent:
            for line in f_ent:
                splits = line.strip().split('\t')
                assert len(splits) == 6
                ent = Entity(splits)
                # corpus -> doc -> entities -> entity
                self.docs[ent.pmid].entities_dict[ent.tid] = ent
                self.docs[ent.pmid].entities_dict1[(ent.start, ent.end, ent.ttype, ent.text)] = ent.tid
                self.docs[ent.pmid].entities.append(ent)

                if self.docs[ent.pmid].text[ent.start: ent.end] != ent.text:
                    print ("txt: \"%s\" -> annot:\"%s\"\t\t%s" % \
                          (self.docs[ent.pmid].text[ent.start: ent.end],
                           ent.text,
                           line))

        if not self.is_test:
            with codecs.open(self.rel_path, encoding=ENCODING) as f_rel:
                for line in f_rel:
                    splits = line.strip().split('\t')

                    if len(splits) == 4:
                        pmid = splits[0]
                        arg1 = self.docs[pmid].entities_dict[splits[2][5:]]
                        arg2 = self.docs[pmid].entities_dict[splits[3][5:]]
                        rtype = splits[1]
                        rel = Relation(pmid, arg1, arg2, rtype)
                        self.docs[rel.pmid].relations.append(rel)
                        #self.docs[rel.pmid].relations_dict[rel.arg1][rel.arg2] = rel

                    else:
                        print ("annot format error: " + line)
