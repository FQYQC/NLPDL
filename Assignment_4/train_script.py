import json
import os.path as osp
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--r', action='store_true')
args = parser.parse_args()

if args.r:
    files = os.listdir('jsons')
    for file in files:
        os.remove('jsons/' + file)
    os.removedirs('jsons')
    os.mkdir('jsons')

    models = ['roberta', 'bert', 'sci_bert']
    model_names = ['roberta-base', 'bert-base-uncased',
                'allenai/scibert_scivocab_uncased']
    datas = ['acl', 'agnews', 'rest']
    data_names = ['acl_sup', 'agnews_sup', 'restaurant_sup']
    use_adapter = [True, False]

    for i, model in enumerate(models):
        for j, data in enumerate(datas):
            for k, adapter in enumerate(use_adapter):
                json_name = model + '_' + data
                if adapter:
                    json_name += '_adapter'
                json_name += '.json'
                json_path = osp.join('jsons', json_name)
                with open(json_path, 'w') as f:
                    json.dump({
                        "report_to": "wandb",
                        "seed": 2022,
                        "model_name_or_path": model_names[i],
                        "dataset_name": data_names[j],
                        "output_dir": "outputs/" + json_name[:-5],
                        "do_train": True,
                        "do_eval": True,
                        "overwrite_output_dir": True,
                        "num_train_epochs": 50,
                        "per_device_train_batch_size": 128,
                        "evaluation_strategy": "epoch",
                        "use_adapter": adapter,
                        "adapter_size": 128
                    }, f)

for json_file in os.listdir('jsons'):
    os.system('python train.py jsons/' + json_file)

