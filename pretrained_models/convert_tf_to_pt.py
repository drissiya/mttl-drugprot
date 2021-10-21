from pytorch_pretrained_bert  import BertConfig, BertForPreTraining
from argparse import ArgumentParser
import tensorflow as tf 
import re
import torch
import numpy as np


def convert(args):
    init_vars = tf.train.list_variables(args.tf_path_model)
    excluded = ['BERTAdam','_power','global_step']
    init_vars = list(filter(lambda x:all([True if e not in x[0] else False for e in excluded]),init_vars))

    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(args.tf_path_model, name)
        names.append(name)
        arrays.append(array)  

    config = BertConfig.from_json_file(args.config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    for name, array in zip(names, arrays):
        name = name.split('/')
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)

    print("Save PyTorch model to {}".format(args.model_type))
    params = {'state':model.state_dict(), 'config': config.to_dict(), 'multi_gpu_on':False}
    torch.save(params,args.pytorch_file)



if __name__ == '__main__':
    parser = ArgumentParser(description="Convert TF BERT-based models to pytorch version")
    parser.add_argument('--tf-path-model', type=str, default='bert/bert_model.ckpt')
    parser.add_argument('--config-file', type=str, default='bert/bert_config.json')
    parser.add_argument('--model-type', type=str, default='bert')
    parser.add_argument('--pytorch-file', type=str, default='bert_base_uncased.pt')

    args = parser.parse_args()
    convert(args)


