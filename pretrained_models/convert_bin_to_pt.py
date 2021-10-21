import torch
from argparse import ArgumentParser
from pytorch_pretrained_bert import BertTokenizer, BertConfig


def convert(args):
    state_dict = torch.load(args.bin_model_path)
    config = BertConfig.from_json_file(args.config_file)
    params = {'state':state_dict, 'config': config.to_dict(), 'multi_gpu_on':False}
    torch.save(params,args.pytorch_file)


if __name__ == '__main__':
    parser = ArgumentParser(description="Convert bin BERT-based models to pytorch version")
    parser.add_argument('--bin-model-path', type=str, default='pytorch_model.bin')
    parser.add_argument('--config-file', type=str, default='config.json')
    parser.add_argument('--model-type', type=str, default='bert')
    parser.add_argument('--pytorch-file', type=str, default='pubmedbert_base_uncased.pt')

    args = parser.parse_args()
    convert(args)


