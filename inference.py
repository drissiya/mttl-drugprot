import os
from sys import path
path.append(os.getcwd())
from dp_extraction.extract import drugprot_extract_guess_relation
from dp_extraction.utils import write_brat_files, get_predicted_sequence_2, get_label_mappers

import codecs
import pickle as pkl
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description="DrugProt extraction")
    parser.add_argument('--dataset', type=str, default='drugprot')
    parser.add_argument('--predicted_json_path', type=str, default='checkpoint/drugprot_test_scores_7.json')
    parser.add_argument('--gold_json_path', type=str, default='data/canonical_data/bert_uncased_lower/drugprot_test.json')
    parser.add_argument('--gold_dir', type=str, default='data/drugprot/test_background')
    parser.add_argument('--guess_file', type=str, default='drugprot_test_predictions_mt_pubmedbert_data_more.tsv')
    args = parser.parse_args()
    label_mappers = get_label_mappers()

    if args.dataset=='drugprot':
        drugprot_guess = get_predicted_sequence_2(label_mappers=label_mappers[4],
                                                  dataset=args.dataset,
                                                  predicted_json_path=args.predicted_json_path, 
                                                  gold_json_path=args.gold_json_path)
        dict_relation = drugprot_extract_guess_relation(guess=drugprot_guess, 
                                                        segment='BIO', 
                                                        gold_dir=args.gold_dir)
        write_brat_files(dict_relations=dict_relation, 
                         tsv_pred_file=args.guess_file, 
                         test_path=args.gold_dir)
        



