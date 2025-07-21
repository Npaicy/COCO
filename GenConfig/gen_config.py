import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
class GenConfig:
    db_config = {
            'database': 'imdb',
            'user': '',
            'password': '',
            'port': 1250,
            'host': ''
        }
    base_db_config = db_config
    remote_db_config_list = [
        base_db_config.copy(),
    ]
    databases = ['basketball', 'walmart','financial', 'movielens',
                'carcinogenesis','accidents', 'tournament',
                'employee', 'geneea', 'genome','seznam', 'fhnk', 'consumer',
                'ssb', 'hepatitis', 'credit','chembl','ergastf1','grants',
                'legalacts','sap','talkingdata','baseball', 'tpch','stats','imdb']
    test_databases = ['stats','imdb','tpch','baseball']
    
    N_bins = 10
    ckpt_dir = './ckpt'
    pretrain_ckpt_dir = './ckpt/pretrain'
    column_stats_dir = './features/dataset/'  
    meta_info_path = './features/meta/metaInfo.pkl'
    assindex_path = './features/meta/assindex.pkl'
    column_path = './features/meta/column.pkl'
    table_path = './features/meta/table.pkl'
    database_path = './features/meta/database.pkl'
    sample_dir = './features/sample'
    distribution_dir = './features/distribution'
    attrEmbed_dir = './features/embedding'
    schema_dir = './features/schema'