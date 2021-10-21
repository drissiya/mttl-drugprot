import os
import argparse
import pickle as pkl
from sys import path
path.append(os.getcwd())
from data_utils.task_def import DataFormat
from data_utils.tac.tac_corpus import TAC, split_data_sequence_tac
from data_utils.n2c2.n2c2_corpus import N2C2, split_data_sequence_n2c2
from data_utils.drugprot.drugprot_data import DrugProt
from data_utils.chemprot.chemprot_data import ChemProt
from data_utils.log_wrapper import create_logger
from experiments.utils import *
from experiments.ADE_utils import *
from experiments.common_utils import dump_rows
logger = create_logger(__name__, to_disk=True, log_file='bert_ner_data_proc_512_cased.log')

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing datasets.')
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--output_dir', type=str, default="data/canonical_data")
    args = parser.parse_args()
    return args

def main(args):
    data_dir = args.data_dir
    data_dir = os.path.abspath(data_dir)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        
    print ("Load TAC training set") 
    TAC_train_temp = TAC(os.path.join(data_dir, "TAC", "TR"), os.path.join(data_dir, "TAC", "train_xml")) 
    TAC_train_temp.load_corpus()    
    TAC_train, TAC_dev = split_data_sequence_tac(TAC_train_temp)
    
    print ("Load TAC test set")
    TAC_test = TAC(os.path.join(data_dir, "TAC", "TE"), os.path.join(data_dir, "TAC", "gold_xml")) 
    TAC_test.load_corpus()
	
    print ("Load n2c2 training set")
    n2c2_train_temp = N2C2(label_dir=data_dir+"/N2C2/training_20180910/")
    n2c2_train_temp.load_corpus()    
    n2c2_train, n2c2_dev = split_data_sequence_n2c2(n2c2_train_temp)
    
    print ("Load n2c2 test set")
    n2c2_test = N2C2(label_dir=data_dir+"/N2C2/test/")
    n2c2_test.load_corpus() 
    
    print ("Load drugprot dataset")
    train_drugprot = DrugProt(data_dir=data_dir+"/drugprot/drugprot_training", segment="BIO")
    train_drugprot.load_corpus()
    dev_drugprot = DrugProt(data_dir=data_dir+"/drugprot/drugprot_development", segment="BIO")
    dev_drugprot.load_corpus()
    test_drugprot = DrugProt(data_dir=data_dir+"/drugprot/test_background", segment="BIO", test=True)
    test_drugprot.load_corpus()
    
    print ("Load chemprot dataset")
    train_chemprot = ChemProt(data_dir=data_dir+"/chemprot/chemprot_training", segment="BIO")
    train_chemprot.load_corpus()
    dev_chemprot = ChemProt(data_dir=data_dir+"/chemprot/chemprot_development", segment="BIO")
    dev_chemprot.load_corpus()
    test_chemprot = ChemProt(data_dir=data_dir+"/chemprot/chemprot_test", segment="BIO", test=True)
    test_chemprot.load_corpus()
    
    train_tac_ner = load_adr_ner(TAC_train)
    print (train_tac_ner[0])
    dev_tac_ner = load_adr_ner(TAC_dev)
    print (dev_tac_ner[0])
    test_tac_ner = load_adr_ner(TAC_test)
    print (test_tac_ner[0])

    logger.info('Loaded {} TAC NER rel train samples'.format(len(train_tac_ner)))
    logger.info('Loaded {} TAC NER rel dev samples'.format(len(dev_tac_ner)))
    logger.info('Loaded {} TAC NER rel test samples'.format(len(test_tac_ner)))
        

    train_tac_relation = load_adr_relation(TAC_train)
    print (train_tac_relation[0])
    dev_tac_relation = load_adr_relation(TAC_dev)
    print (dev_tac_relation[0])
    test_tac_relation = load_adr_relation(TAC_test)
    print (test_tac_relation[0])

    logger.info('Loaded {} TAC Relation train samples'.format(len(train_tac_relation)))
    logger.info('Loaded {} TAC Relation dev samples'.format(len(dev_tac_relation)))
    logger.info('Loaded {} TAC Relation test samples'.format(len(test_tac_relation)))
    
	
    train_n2c2_ner = load_adr_ner(n2c2_train)
    print (train_n2c2_ner[0])
    dev_n2c2_ner = load_adr_ner(n2c2_dev)
    print (dev_n2c2_ner[0])
    test_n2c2_ner = load_adr_ner(n2c2_test)
    print (test_n2c2_ner[0])

    logger.info('Loaded {} n2c2 NER train samples'.format(len(train_n2c2_ner)))
    logger.info('Loaded {} n2c2 NER dev samples'.format(len(dev_n2c2_ner)))
    logger.info('Loaded {} n2c2 NER test samples'.format(len(test_n2c2_ner)))
	
    train_n2c2_relation = load_adr_relation(n2c2_train)
    print (train_n2c2_relation[0])
    dev_n2c2_relation = load_adr_relation(n2c2_dev)
    print (dev_n2c2_relation[0])
    test_n2c2_relation = load_adr_relation(n2c2_test)
    print (test_n2c2_relation[0])

    logger.info('Loaded {} n2c2 Relation train samples'.format(len(train_n2c2_relation)))
    logger.info('Loaded {} n2c2 Relation dev samples'.format(len(dev_n2c2_relation)))
    logger.info('Loaded {} n2c2 Relation test samples'.format(len(test_n2c2_relation)))
	

    train_drugprot_relation = load_adr_relation(train_drugprot)
    print (train_drugprot_relation[0])
    dev_drugprot_relation = load_adr_relation(dev_drugprot)
    print (dev_drugprot_relation[0])
    test_drugprot_relation = load_adr_relation(test_drugprot)
    print (test_drugprot_relation[0])

    logger.info('Loaded {} drugprot train samples'.format(len(train_drugprot_relation)))
    logger.info('Loaded {} drugprot dev samples'.format(len(dev_drugprot_relation)))
    logger.info('Loaded {} drugprot test samples'.format(len(test_drugprot_relation)))
    
    train_chemprot_relation = load_adr_relation(train_chemprot)
    print (train_chemprot_relation[0])
    dev_chemprot_relation = load_adr_relation(dev_chemprot)
    print (dev_chemprot_relation[0])
    test_chemprot_relation = load_adr_relation(test_chemprot)
    print (test_chemprot_relation[0])

    logger.info('Loaded {} chemprot train samples'.format(len(train_chemprot_relation)))
    logger.info('Loaded {} chemprot dev samples'.format(len(dev_chemprot_relation)))
    logger.info('Loaded {} chemprot test samples'.format(len(test_chemprot_relation)))
	
    train_i2b2_concept = load_i2b2_2010_concept(data_dir + '/i2b2/train/beth/', data_dir + '/i2b2/train/partners/')
    test_i2b2_concept = load_i2b2_2010_concept_test(data_dir + '/i2b2/test/')
    print (train_i2b2_concept[0])
    print (test_i2b2_concept[0])
    logger.info('Loaded {} i2b2 concepts train samples'.format(len(train_i2b2_concept)))
    logger.info('Loaded {} i2b2 concepts test samples'.format(len(test_i2b2_concept)))

    train_i2b2_relation = load_i2b2_2010_relation(data_dir + '/i2b2/train/beth/', data_dir + '/i2b2/train/partners/')
    test_i2b2_relation = load_i2b2_2010_relation_test(data_dir + '/i2b2/test/')
    print (train_i2b2_relation[0])
    print (test_i2b2_relation[0])
    logger.info('Loaded {} i2b2 relations train samples'.format(len(train_i2b2_relation)))
    logger.info('Loaded {} i2b2 relations test samples'.format(len(test_i2b2_relation)))

    train_path = os.path.join(data_dir, 'i2b2_2009/train.txt')
    dev_path = os.path.join(data_dir, 'i2b2_2009/valid.txt')
    test_path = os.path.join(data_dir, 'i2b2_2009/test.txt')

    train_data_i2b2 = load_conll_ner(train_path)
    dev_data_i2b2 = load_conll_ner(dev_path)
    test_data_i2b2 = load_conll_ner(test_path)
    print (train_data_i2b2[0])
    print (dev_data_i2b2[0])
    print (test_data_i2b2[0])
    logger.info('Loaded {} NER i2b2 train samples'.format(len(train_data_i2b2)))
    logger.info('Loaded {} NER i2b2 dev samples'.format(len(dev_data_i2b2)))
    logger.info('Loaded {} NER i2b2 test samples'.format(len(test_data_i2b2)))

	
    X_train, X_test, y_train, y_test = list_of_sentence(data_dir + "/ADE-Corpus/")
    train_ade = load_ade_corpus(X_train, y_train)
    test_ade = load_ade_corpus(X_test, y_test)
    print (train_ade[0])
    print (test_ade[0])
    logger.info('Loaded {} ADE corpus train samples'.format(len(train_ade)))
    logger.info('Loaded {} ADE corpus test samples'.format(len(test_ade)))
	
    train_path = os.path.join(data_dir, 'BC2GM/train.tsv')
    dev_path = os.path.join(data_dir, 'BC2GM/devel.tsv')
    test_path = os.path.join(data_dir, 'BC2GM/test.tsv')


    train_data_bc2gm = load_ner(train_path)
    dev_data_bc2gm = load_ner(dev_path)
    test_data_bc2gm = load_ner(test_path)
    print (train_data_bc2gm[0])
    print (dev_data_bc2gm[0])
    print (test_data_bc2gm[0])
    logger.info('Loaded {} BC2GM train samples'.format(len(train_data_bc2gm)))
    logger.info('Loaded {} BC2GM dev samples'.format(len(dev_data_bc2gm)))
    logger.info('Loaded {} BC2GM test samples'.format(len(test_data_bc2gm)))

    train_path = os.path.join(data_dir, 'BC4CHEMD/train.tsv')
    dev_path = os.path.join(data_dir, 'BC4CHEMD/devel.tsv')
    test_path = os.path.join(data_dir, 'BC4CHEMD/test.tsv')


    train_data_bc4chemd = load_ner(train_path)
    dev_data_bc4chemd = load_ner(dev_path)
    test_data_bc4chemd = load_ner(test_path)
    print (train_data_bc4chemd[0])
    print (dev_data_bc4chemd[0])
    print (test_data_bc4chemd[0])
    logger.info('Loaded {} BC4CHEMD train samples'.format(len(train_data_bc4chemd)))
    logger.info('Loaded {} BC4CHEMD dev samples'.format(len(dev_data_bc4chemd)))
    logger.info('Loaded {} BC4CHEMD test samples'.format(len(test_data_bc4chemd)))

    train_path = os.path.join(data_dir, 'BC5CDR/train.tsv')
    dev_path = os.path.join(data_dir, 'BC5CDR/devel.tsv')
    test_path = os.path.join(data_dir, 'BC5CDR/test.tsv')


    train_data_bc5cdr = load_ner(train_path)
    dev_data_bc5cdr = load_ner(dev_path)
    test_data_bc5cdr = load_ner(test_path)
    print (train_data_bc5cdr[0])
    print (dev_data_bc5cdr[0])
    print (test_data_bc5cdr[0])
    logger.info('Loaded {} BC5CDR train samples'.format(len(train_data_bc5cdr)))
    logger.info('Loaded {} BC5CDR dev samples'.format(len(dev_data_bc5cdr)))
    logger.info('Loaded {} BC5CDR test samples'.format(len(test_data_bc5cdr)))

    train_path = os.path.join(data_dir, 'NCBI-disease/train.tsv')
    dev_path = os.path.join(data_dir, 'NCBI-disease/devel.tsv')
    test_path = os.path.join(data_dir, 'NCBI-disease/test.tsv')


    train_data_ncbi = load_ner(train_path)
    dev_data_ncbi = load_ner(dev_path)
    test_data_ncbi = load_ner(test_path)
    print (train_data_ncbi[0])
    print (dev_data_ncbi[0])
    print (test_data_ncbi[0])
    logger.info('Loaded {} NCBI-disease train samples'.format(len(train_data_ncbi)))
    logger.info('Loaded {} NCBI-disease dev samples'.format(len(dev_data_ncbi)))
    logger.info('Loaded {} NCBI-disease test samples'.format(len(test_data_ncbi)))

    train_path = os.path.join(data_dir, 'JNLPBA/train.tsv')
    dev_path = os.path.join(data_dir, 'JNLPBA/devel.tsv')
    test_path = os.path.join(data_dir, 'JNLPBA/test.tsv')


    train_data_jnlpba = load_ner(train_path)
    dev_data_jnlpba = load_ner(dev_path)
    test_data_jnlpba = load_ner(test_path)
    print (train_data_jnlpba[0])
    print (dev_data_jnlpba[0])
    print (test_data_jnlpba[0])
    logger.info('Loaded {} JNLPBA train samples'.format(len(train_data_jnlpba)))
    logger.info('Loaded {} JNLPBA dev samples'.format(len(dev_data_jnlpba)))
    logger.info('Loaded {} JNLPBA test samples'.format(len(test_data_jnlpba)))


    bert_root = args.output_dir
    if not os.path.isdir(bert_root):
        os.mkdir(bert_root)
	


    train_fout = os.path.join(bert_root, 'tacsource_train.tsv')
    dev_fout = os.path.join(bert_root, 'tacsource_dev.tsv')
    test_fout = os.path.join(bert_root, 'tacsource_test.tsv')

    dump_rows(train_tac_ner, train_fout, DataFormat.Seqence)
    dump_rows(dev_tac_ner, dev_fout, DataFormat.Seqence)
    dump_rows(test_tac_ner, test_fout, DataFormat.Seqence)
    logger.info('done with TAC source')
	

    train_fout = os.path.join(bert_root, 'tacrelation_train.tsv')
    dev_fout = os.path.join(bert_root, 'tacrelation_dev.tsv')
    test_fout = os.path.join(bert_root, 'tacrelation_test.tsv')
    
    dump_rows(train_tac_relation, train_fout, DataFormat.Seqence)
    dump_rows(dev_tac_relation, dev_fout, DataFormat.Seqence)
    dump_rows(test_tac_relation, test_fout, DataFormat.Seqence)
    logger.info('done with TAC Relation')
	
    train_fout = os.path.join(bert_root, 'n2c2source_train.tsv')
    dev_fout = os.path.join(bert_root, 'n2c2source_dev.tsv')
    test_fout = os.path.join(bert_root, 'n2c2source_test.tsv')

    dump_rows(train_n2c2_ner, train_fout, DataFormat.Seqence)
    dump_rows(dev_n2c2_ner, dev_fout, DataFormat.Seqence)
    dump_rows(test_n2c2_ner, test_fout, DataFormat.Seqence)
    logger.info('done with n2c2 source')
	

    train_fout = os.path.join(bert_root, 'n2c2relation_train.tsv')
    dev_fout = os.path.join(bert_root, 'n2c2relation_dev.tsv')
    test_fout = os.path.join(bert_root, 'n2c2relation_test.tsv')
    
    dump_rows(train_n2c2_relation, train_fout, DataFormat.Seqence)
    dump_rows(dev_n2c2_relation, dev_fout, DataFormat.Seqence)
    dump_rows(test_n2c2_relation, test_fout, DataFormat.Seqence)
    logger.info('done with n2c2 Relation')
    
    train_fout = os.path.join(bert_root, 'drugprot_train.tsv')
    dev_fout = os.path.join(bert_root, 'drugprot_dev.tsv')
    test_fout = os.path.join(bert_root, 'drugprot_test.tsv')
    
    dump_rows(train_drugprot_relation, train_fout, DataFormat.Seqence)
    dump_rows(dev_drugprot_relation, dev_fout, DataFormat.Seqence)
    dump_rows(test_drugprot_relation, test_fout, DataFormat.Seqence)
    logger.info('done with drugprot')

    train_fout = os.path.join(bert_root, 'chemprot_train.tsv')
    dev_fout = os.path.join(bert_root, 'chemprot_dev.tsv')
    test_fout = os.path.join(bert_root, 'chemprot_test.tsv')
    
    dump_rows(train_chemprot_relation, train_fout, DataFormat.Seqence)
    dump_rows(dev_chemprot_relation, dev_fout, DataFormat.Seqence)
    dump_rows(test_chemprot_relation, test_fout, DataFormat.Seqence)
    logger.info('done with chemprot')
	
    train_fout = os.path.join(bert_root, 'i2b2concept_train.tsv')
    test_fout = os.path.join(bert_root, 'i2b2concept_test.tsv')

    dump_rows(train_i2b2_concept, train_fout, DataFormat.Seqence)
    dump_rows(test_i2b2_concept, test_fout, DataFormat.Seqence)
    logger.info('done with i2b2 concepts')
	
    train_fout = os.path.join(bert_root, 'i2b2relation_train.tsv')
    test_fout = os.path.join(bert_root, 'i2b2relation_test.tsv')

    dump_rows(train_i2b2_relation, train_fout, DataFormat.PremiseOnly)
    dump_rows(test_i2b2_relation, test_fout, DataFormat.PremiseOnly)
    logger.info('done with i2b2 relations')
	
	
    train_fout = os.path.join(bert_root, 'i2b2ner2009_train.tsv')
    dev_fout = os.path.join(bert_root, 'i2b2ner2009_dev.tsv')
    test_fout = os.path.join(bert_root, 'i2b2ner2009_test.tsv')

    dump_rows(train_data_i2b2, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_i2b2, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_i2b2, test_fout, DataFormat.Seqence)
    logger.info('done with NER i2b2')

	
    train_fout = os.path.join(bert_root, 'ade_train.tsv')
    test_fout = os.path.join(bert_root, 'ade_test.tsv')

    dump_rows(train_ade, train_fout, DataFormat.PremiseOnly)
    dump_rows(test_ade, test_fout, DataFormat.PremiseOnly)
    logger.info('done with ADE corpus')
	
    train_fout = os.path.join(bert_root, 'bc2gm_train.tsv')
    dev_fout = os.path.join(bert_root, 'bc2gm_dev.tsv')
    test_fout = os.path.join(bert_root, 'bc2gm_test.tsv')

    dump_rows(train_data_bc2gm, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_bc2gm, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_bc2gm, test_fout, DataFormat.Seqence)
    logger.info('done with bc2gm')
	
    train_fout = os.path.join(bert_root, 'bc4chemd_train.tsv')
    dev_fout = os.path.join(bert_root, 'bc4chemd_dev.tsv')
    test_fout = os.path.join(bert_root, 'bc4chemd_test.tsv')

    dump_rows(train_data_bc4chemd, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_bc4chemd, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_bc4chemd, test_fout, DataFormat.Seqence)
    logger.info('done with bc4chemd')

    train_fout = os.path.join(bert_root, 'bc5cdr_train.tsv')
    dev_fout = os.path.join(bert_root, 'bc5cdr_dev.tsv')
    test_fout = os.path.join(bert_root, 'bc5cdr_test.tsv')

    dump_rows(train_data_bc5cdr, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_bc5cdr, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_bc5cdr, test_fout, DataFormat.Seqence)
    logger.info('done with bc5cdr')

    train_fout = os.path.join(bert_root, 'ncbi_train.tsv')
    dev_fout = os.path.join(bert_root, 'ncbi_dev.tsv')
    test_fout = os.path.join(bert_root, 'ncbi_test.tsv')

    dump_rows(train_data_ncbi, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_ncbi, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_ncbi, test_fout, DataFormat.Seqence)
    logger.info('done with ncbi')
	
    train_fout = os.path.join(bert_root, 'jnlpba_train.tsv')
    dev_fout = os.path.join(bert_root, 'jnlpba_dev.tsv')
    test_fout = os.path.join(bert_root, 'jnlpba_test.tsv')

    dump_rows(train_data_jnlpba, train_fout, DataFormat.Seqence)
    dump_rows(dev_data_jnlpba, dev_fout, DataFormat.Seqence)
    dump_rows(test_data_jnlpba, test_fout, DataFormat.Seqence)
    logger.info('done with jnlpba')
	
if __name__ == '__main__':
    args = parse_args()
    main(args)