import yaml
import json
import os
from data_utils.drugprot.brat import Corpus
from data_utils.task_def import TaskType, DataFormat
from data_utils.drugprot.drugprot_data import DrugProt
from experiments.exp_def import TaskDefs

ENCODING = 'utf-8'

def get_label_mappers(path_yml="data_task_def.yml"):
    task_defs = TaskDefs(path_yml)
    task_def_dic = yaml.safe_load(open(path_yml))
    label_mappers = []
    for task, task_def in task_def_dic.items():
        data_format = DataFormat[task_def["data_format"]]
        task_type = TaskType[task_def["task_type"]]
        label_mapper = task_defs.global_map.get(task, None)
        label_mappers.append(label_mapper)
    return label_mappers 
	
def trim_sequence(prediction, true_set, label_mappers):
    predict_lines = []
    for pred, true in zip(prediction, true_set):
        p_label = []
        for p, t in zip(pred, true):
            l= label_mappers.ind2tok[p]
            tr= label_mappers.ind2tok[t]
            if tr == 'X': continue
            if l == 'CLS': continue
            if l == 'SEP': continue
            if l == 'X': l = 'O'
            p_label.append(l)
        predict_lines.append(p_label)
    return predict_lines


def write_brat_files(dict_relations, tsv_pred_file, test_path):
    file = open(tsv_pred_file,"w", encoding=ENCODING)
    corpus = Corpus(test_path, is_test=True)
    for d in corpus.doc_ids:
        if len(dict_relations[d])>0:
            for m in dict_relations[d]:
                #file.write(m[0] + '\t' + m[3] + '\t' + ' Arg1:' + m[1] + '\t' + ' Arg2:' + m[2] + '\n')
                file.write(m[0] + '\t' + m[3] + '	Arg1:' + m[1] + '	Arg2:' + m[2] + '\n')
    file.close()       
	
	
def get_predicted_sequence_2(label_mappers,
                             dataset,
                             predicted_json_path, 
                             gold_json_path):

    if dataset=='drugprot':
        data_test = DrugProt() 
    with open(predicted_json_path) as json_file:
        data = json.load(json_file)
        prediction = data["predictions"]

    data = [json.loads(line) for line in open(gold_json_path, 'r')]
    for p in data:
        data_test.t_sentence_relation.append(p["tokens"])
        data_test.t_toks_relation.append(p["tokens"])
        data_test.t_segment_relation.append(p["label"])
        data_test.t_start_relation.append(p["start"])
        data_test.t_section_relation.append(p["section"])
        data_test.t_len_relation.append(p["lenn"])
        data_test.t_ade.append(p["ade"])
        data_test.t_modifiers.append(p["modifier"])
        data_test.t_drug_relation.append(p["drug"])
    predicted_sequence = trim_sequence(prediction, data_test.t_segment_relation, label_mappers)

    data_test.t_segment_relation = predicted_sequence

    return data_test