import os.path as osp
import json
import datasets
from datasets import Dataset, load_dataset
from datasets.dataset_dict import DatasetDict


def get_dataset(dataset_name: str, sep_token: str = '<sep>'):
    '''
        dataset_name: str, the name of the dataset
        sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')

        few_shot: For tasks with fewer than 5 labels, we construct train 
                  and dev with 32 samples from the original training data 
                  and ensure the number of labels is balanced. For tasks 
                  with more than 5 labels like TNews and YahooAnswer, we 
                  randomly select 8 samples for each label.
    '''

    def get_dataset_dict(dataset_name, label_offset=0):
        dataset = None
        filepath = None
        data_type = None
        data_qty = 'sup'
        label_qty = None
        if dataset_name[:10] == 'restaurant':
            filepath = 'SemEval14-res'
            data_type = 'SemEval14'
            label_qty = '3'
        elif dataset_name[:6] == 'laptop':
            filepath = 'SemEval14-laptop'
            data_type = 'SemEval14'
            label_qty = '3'
        elif dataset_name[:3] == 'acl':
            filepath = 'acl-arc'
            data_type = 'ACL-ARC'
            label_qty = '6'
        elif dataset_name[:6] == 'agnews':
            data_type = 'ag_news'
            label_qty = '4'

        if dataset_name[-2:] == 'fs':
            data_qty = 'fs'

        if data_type == 'SemEval14':
            filepath = osp.join('mydatasets', filepath)
            pol2lab = {'positive': 0, 'negative': 1, 'neutral': 2}
            with open(osp.join(filepath, 'train.json'), 'r') as f:
                train_data = json.load(f)
            with open(osp.join(filepath, 'test.json'), 'r') as f:
                test_data = json.load(f)
            train_data = {
                'text': [item['sentence'] for (_, item) in train_data.items()],
                'label': [pol2lab[item['polarity']]+label_offset for (_, item) in train_data.items()]
            }
            test_data = {
                'text': [item['sentence'] for (_, item) in test_data.items()],
                'label': [pol2lab[item['polarity']]+label_offset for (_, item) in test_data.items()]
            }

        elif data_type == 'ACL-ARC':
            filepath = osp.join('mydatasets', filepath)
            intent2lab = {'Background':0, 'CompareOrContrast':1,'Future':2,'Uses':3,'Motivation':4,'Extends':5,}
            with open(osp.join(filepath, 'train.jsonl'), 'r') as f:
                train_data = [json.loads(line) for line in f.readlines()]
            with open(osp.join(filepath, 'test.jsonl'), 'r') as f:
                test_data = [json.loads(line) for line in f.readlines()]

            train_data = {
                'text': [item['text'] for item in train_data],
                'label': [intent2lab[item['intent']]+label_offset for item in train_data]
            }
            test_data = {
                'text': [item['text'] for item in test_data],
                'label': [intent2lab[item['intent']]+label_offset for item in test_data]
            }

        elif data_type == 'ag_news':
            train_data = datasets.load_dataset('ag_news', split='test')
            dataset = train_data.train_test_split(test_size=0.1, seed=2022)
            train_data = {
                'text': train_data['text'],
                'label': [item+label_offset for item in train_data['label']]
            }
            test_data = {
                'text': dataset['test']['text'],
                'label': [item+label_offset for item in dataset['test']['label']]
            }

        dataset = {'train': train_data, 'test': test_data}
        if data_qty == 'fs':
            if data_type == 'ACL-ARC':
                train_text = []
                train_label = []
                for i in range(6):
                    idx = [j for j, item in enumerate(train_data['label']) if item == i]
                    train_text += [train_data['text'][j] for j in idx[:8]]
                    train_label += [train_data['label'][j] for j in idx[:8]]
            else:
                dataset['train']['text'] = dataset['train']['text'][:32]
                dataset['train']['label'] = dataset['train']['label'][:32]
            # TODO:!!!
        return dataset, label_qty

    if isinstance(dataset_name, str):
        dataset_name = [dataset_name]

    tot_label = 0
    for (i, dataset_name_i) in enumerate(dataset_name):
        dataset_i, label_qty = get_dataset_dict(dataset_name_i, tot_label)
        if i == 0:
            dataset = dataset_i
        else:
            for sp1 in ['train', 'test']:
                for sp2 in ['text', 'label']:
                    dataset[sp1][sp2].extend(dataset_i[sp1][sp2])
        tot_label += int(label_qty)

    dataset = DatasetDict({'train': Dataset.from_dict(
        dataset['train']), 'test': Dataset.from_dict(dataset['test'])})

    return dataset


if __name__ == '__main__':
    dataset = get_dataset(['restaurant_fs', 'agnews_fs'], '<sep>')
    print(dataset)
