import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LQO.pghelper import PGHelper
import pickle
import json
from GenConfig.gen_config import GenConfig
import copy
from LQO.constant import PGDATATYPE, PGDATETYPE, PGCHARTYPE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def get_meta_info(database_names):
    # Database configurations
    if os.path.exists(GenConfig.meta_info_path):
        old_metaInfo = pickle.load(open(GenConfig.meta_info_path, 'rb'))
    else:
        old_metaInfo = None
    pg_helper = PGHelper(GenConfig.db_config)
    # Initialize metaInfo dictionary
    metaInfo = {db_name: {} for db_name in database_names}
    columnType2Idx = {}
    
    # Get statistics and column types for each database
    for db_name in database_names:
        print(db_name,flush=True)
        pg_helper.reconnect(db_name)
        metaInfo[db_name] = pg_helper.get_statistics(old_metaInfo[db_name])
        columnType = pg_helper.get_column_type()
        # Convert column types to indices
        for colName, columnType in columnType.items():
            # if columnType not in columnType2Idx:
            #     columnType2Idx[columnType] = len(columnType2Idx) + 1
            if columnType in ['smallint','integer','bigint','smallserial','serial','bigserial']:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 1
            elif columnType in ['decimal','numeric','real','double precision']:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 2
            else:
                metaInfo[db_name]['colAttr'][colName]['colDtype'] = 3

    # post processing
    max_avg_width = 0
    # max_correlation = 0
    max_table_size = 0
    max_num_columns = 0
    # max_num_references = 0
    max_num_indexes = 0
    for db_name in database_names:
        for col_name,col_info in metaInfo[db_name]['colAttr'].items():
            if math.log2(col_info['avg_width'] + 1) > max_avg_width:
                max_avg_width = math.log2(col_info['avg_width'] + 1)
        for table_name,table_info in metaInfo[db_name]['tableAttr'].items():
            if table_info['table_size'] > max_table_size:
                max_table_size = table_info['table_size']
            if table_info['num_columns'] > max_num_columns:
                max_num_columns = table_info['num_columns']
            # if table_info['num_references'] > max_num_references:
            #     max_num_references = table_info['num_references']
            if table_info['num_indexes'] > max_num_indexes:
                max_num_indexes = table_info['num_indexes']

    for db_name in database_names:
        for col_name,col_info in metaInfo[db_name]['colAttr'].items():
            norm_avg_width = math.log2(col_info['avg_width'] + 1) / max_avg_width
            if norm_avg_width > 0.5:
                col_info['avg_width'] = 3
            elif norm_avg_width < 0.25:
                col_info['avg_width'] = 1
            else:
                col_info['avg_width'] = 2
        for table_name,table_info in metaInfo[db_name]['tableAttr'].items():
            norm_table_size = math.log10(table_info['table_size'] + 1) / math.log10(max_table_size + 1)
            norm_num_columns = table_info['num_columns'] / max_num_columns
            norm_num_indexes = table_info['num_indexes'] / max_num_indexes
            metaInfo[db_name]['tableAttr'][table_name]['table_size'] = norm_table_size
            metaInfo[db_name]['tableAttr'][table_name]['num_columns'] = norm_num_columns
            metaInfo[db_name]['tableAttr'][table_name]['num_indexes'] = norm_num_indexes

    pg_helper.close()
    return metaInfo

metaInfo = get_meta_info(GenConfig.databases)

with open(GenConfig.meta_info_path, 'wb') as f:
    pickle.dump(metaInfo, f)