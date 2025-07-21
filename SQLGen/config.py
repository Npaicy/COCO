import os
from GenConfig.gen_config import GenConfig
from SQLGen.meta_types import *
import torch
class Config:
    def __init__(self):
        
        self.db_config = GenConfig.db_config
        self.N_bins = GenConfig.N_bins
        self.databases = GenConfig.databases
        self.remote_db_config_list = GenConfig.remote_db_config_list
        self.test_databases = GenConfig.test_databases
        self.max_value_num = 10
        self.operator_num = len(OPERATORDICT)
        self.aggregator_num = len(AGGREGATORDICT)
        self.logical_operator_num = len(LOGICALOPERATORDICT)
        self.max_column_num = AgentActionType.COL_END - AgentActionType.COL_START
        self.trigger_action_num =  TriggerActionType.AFTER_ONE_PROJECTION.value + 2
        # ======    SQLGen Model    ========
        self.dropout = 0.2
        self.head_size = 6
        self.hidden_dim = 512
        self.ffn_dim = 512
        self.num_layers = 6
        self.device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu") 
        self.model_config = {
            'device': self.device,
            'column': {
                'colDtype': (6, 16),
                'semantic': (16, 16),
                'distribution': (10, 16),
                'meta': (21, 32),  # n_distinct, n_null, is_index, is_primary, is_foreign
                'size':64
            },
            'table': {
                'tableMeta': (3, 8),
                # 'columnIdx': (16, 16),
                'tableEmbed': (16, 16),
                'size':64
            },
            'node':{
                'value': (10, 32),
                'operator': (14, 32),
                'aggregate': (9, 32),
                'cond': (3, 16),
                'predicate': 64, # 5 * 3 + 5 * 3 + 2 * 2
                'final': self.hidden_dim, # 16 + 16 + 10 + 5 = 47

            }
        }
         # self.model_config['height'][1] +
        self.train_params = {
            # SAC parameters
            "action_dim": AgentActionType.COL_END + 1,
            "hidden_dim": self.hidden_dim,
            "buffer_size": 1000000,  
            "batch_size": 128,       
            "gamma": 0.99,
            "tau": 5e-3,             
            "n_step": 4,             # Multi-step returns for better learning
            "prioritized_replay": True,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "actor_lr": 1e-5,        # Separate learning rates
            "critic_lr": 5e-5,
            "entropy_lr": 1e-5,
            "grad_clip": 40.0,       # Gradient clipping for stability
            "layer_norm": True,      # Layer normalization for better training
            "target_network_update_freq": 0,  # Use soft updates (tau)
            "automatic_entropy_tuning": True,
            # "deterministic_eval": True  # Use deterministic policy for eval
        }
        
        self.column_feature_path = GenConfig.column_path
        self.table_feature_path = GenConfig.table_path
        self.checkpoint_dir = GenConfig.ckpt_dir
        self.pretrain_ckpt_dir = GenConfig.pretrain_ckpt_dir
        self.sqlgen_agent_path = os.path.join(self.checkpoint_dir, 'sqlgen_agent')