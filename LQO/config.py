
import os
from LQO.constant import *
from GenConfig.gen_config import GenConfig
import torch

class Config:
    def __init__(self):

        # ======   PG Config   =======
        self.dbConfig = GenConfig.db_config
        self.databases = GenConfig.databases
        self.test_databases = GenConfig.test_databases
        self.N_bins = GenConfig.N_bins
        self.remote_db_config_list = GenConfig.remote_db_config_list
        # ======    LQO Model    ========
        self.dropout = 0.2 #0.2
        self.mlp_dropout = 0.2#0.3 
        self.head_size = 1
        self.ffn_dim = 512
        self.num_layers = 1
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
        self.model_config = {
            'device': self.device,
            'type': (len(TYPE2IDX) + 1, 32),  # (input_dim, embed_dim)
            'pos': (5, 16),
            'dbEst': (2, 16),
            'height': (HEIGHTSIZE, 32),
            'column': {
                'colDtype': (5, 16),
                # 'semantic': (16, 16),
                'distribution': (9, 16),
                'meta': (18 + 3, 16),  # n_distinct, n_null, is_index, is_primary, is_foreign
                'size':16
            },
            # 'filter': {
            #     'op': (len(OP2IDX) + 1, 16),  #operator type
            #     'dtype': (6, 16),   #value type
            #                         # isInMCV,
            #                         # isInHist
            #                         # column
            #                         # value Range
            #     'size':16
            # },
            'join': {'size':16},
            # 'table': {
            #     'tableMeta': (3, 16), # table_size, num_columns, num_references, num_indexes
            #     # 'columnIdx': (16, 16),
            #     # 'tableEmbed': (16, 16),
            #     'size':16
            # }
        }
        self.hidden_dim = 128
        self.node_dim = self.hidden_dim - self.model_config['height'][1] # self.hidden_dim - self.model_config['height'][1]
        self.node_input = 6 + 1 + MAXFILTER * 6 + MAXJOIN * 2
        self.mlp_dim = 128   # The Dimension of RL's MLP 
        self.column_feature_path = GenConfig.column_path
        self.table_feature_path = GenConfig.table_path
        self.assindex_path = GenConfig.assindex_path
        self.meta_info_path = GenConfig.meta_info_path
        self.checkpoint_dir = GenConfig.ckpt_dir
        self.lqo_agent_path = self.checkpoint_dir
        self.pretrain_ckpt_dir = GenConfig.pretrain_ckpt_dir
        self.test_sql_path = ['./TestWorkload/test_job_sql.pkl',
                              './TestWorkload/test_jobext_sql.pkl',
                              './TestWorkload/test_stats_sql.pkl']