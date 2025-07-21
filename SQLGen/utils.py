import json
import os
from types import SimpleNamespace
import pickle
from GenConfig.gen_config import GenConfig

def load_schema_json(dataset):
    schema_path = os.path.join(GenConfig.schema_dir, f'{dataset}.json')
    assert os.path.exists(schema_path), f"Could not find schema.json ({schema_path})"
    return load_json(schema_path)

def load_meta_data(dataset):
    all_table_info = pickle.load(open(GenConfig.table_path, 'rb'))
    all_column_info = pickle.load(open(GenConfig.column_path, 'rb'))
    index_info = pickle.load(open(GenConfig.assindex_path, 'rb'))
    table_info = {}
    column_info = {}
    for table_name in index_info['table2idx'].keys():
        if table_name.split('.')[0] == dataset:
            idx = index_info['table2idx'][table_name]
            table_info[table_name.split('.')[1]] = {'features':{key:all_table_info[key][idx] for key in all_table_info.keys()}, 'include_column':[]}
    for column_name in index_info['column2idx'].keys():
        if column_name.split('.')[0] == dataset:
            idx = index_info['column2idx'][column_name]
            column_info['.'.join(column_name.split('.')[1:])] = {key:all_column_info[key][idx] for key in all_column_info.keys()}
            table_info[column_name.split('.')[1]]['include_column'].append('.'.join(column_name.split('.')[1:]))
    meta_info = {'table':table_info, 'column':column_info}
    return meta_info
def load_index_info():
    index_info = pickle.load(open(GenConfig.assindex_path, 'rb'))

    return index_info
def load_column_statistics(dataset, namespace=True):
    path = os.path.join(GenConfig.column_stats_dir, f'{dataset}_column_statistics.json')
    assert os.path.exists(path), f"Could not find file ({path})"
    return load_json(path, namespace=namespace)


def load_json(path, namespace=True):
    with open(path) as json_file:
        if namespace:
            json_obj = json.load(json_file, object_hook=lambda d: SimpleNamespace(**d))
        else:
            json_obj = json.load(json_file)
    return json_obj


def rand_choice(randstate, l, no_elements=None, replace=False):
    if no_elements is None:
        idx = randstate.randint(0, len(l)) - 1
        return l[idx]
    else:
        idxs = randstate.choice(range(len(l)), no_elements, replace=replace)
        return [l[i] for i in idxs]