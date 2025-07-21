import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import collections
import gymnasium as gym
from gymnasium import spaces
import random as randstate
import copy
from SQLGen.meta_types import *
from SQLGen.utils import load_schema_json, load_column_statistics, rand_choice, load_index_info
from SQLGen.config import Config
config = Config()
class QueryState:
    def __init__(self, dataset, index_info, possible_joins, possible_column, 
                 possible_predicates_type, possible_aggregations, possible_group_by_columns,
                 max_predicates_per_col, max_aggregates_per_col, max_group_by_per_col, max_value_one_predicate):
        self.dataset = dataset
        self.index_info = index_info
        self._possible_joins = possible_joins
        self._possible_column = possible_column
        self._possible_predicates_type = possible_predicates_type
        self._possible_aggregations = possible_aggregations
        self._possible_group_by_columns = possible_group_by_columns

        self.max_predicates_per_col = max_predicates_per_col
        self.max_aggregates_per_col = max_aggregates_per_col
        self.max_group_by_per_col = max_group_by_per_col
        self.max_value_one_predicate = max_value_one_predicate
        self.len_node_vector = 2 + (2 + self.max_value_one_predicate) * self.max_predicates_per_col - 1 + self.max_aggregates_per_col
        
        self.table_to_cols = collections.defaultdict(list)
        for col_name, col_idx in self._possible_column.items():
            table_name = col_name.split('.')[0]
            self.table_to_cols[table_name].append(col_idx)

    def reset(self):
        self.selected_tables = []  
        self.selected_joins = []   # [(t1.c1,t2.c2),...]
        self.selected_predicates = {}  #  [(a,>,1),...]
        self.global_predicates = {}  #  [(a,>,1),...]
        self.selected_projection = []  #  [(t1.c1,count),...]
        self.selected_group_bys = []  # 
        self.having_clause = None  # 
        self.limit = None  # 
        self.is_complete = False  # 
        self.current_possible_joins = copy.deepcopy(self._possible_joins)
        self.current_predicates_column = []
        self.current_projection_column = []
        self.numeric_predicate_fild = {}
        self.baseline_state = {}  
        self.vector_state = {'predicates': {}, 'projection': {}, 'group_by': {}}  # state vector for later processing
        for column in self._possible_column:
            table_name = self.dataset + '.' + column.split('.')[0]
            column_name = self.dataset + '.' + column
            self.baseline_state[column] = np.zeros(self.len_node_vector, dtype=np.int16)
            self.baseline_state[column][0] = self.index_info['table2idx'][table_name]
            self.baseline_state[column][1] = self.index_info['column2idx'][column_name]
        
    def update_joins(self, unit, mode):
        if mode == 'left':
            self.selected_joins.append([unit])
        elif mode == 'right':
            self.selected_joins[-1].append(unit)
            self.delete_possible_joins(self.selected_joins[-1])
            self.update_global_predicates_keys(self.selected_joins[-1])
        table, column = unit.split('.')
        if table not in self.selected_tables:
            self.selected_tables.append(table)

    def update_predicates(self, unit, mode, ori_value = None):
        if mode == 'column':
            if unit not in self.selected_predicates:
                self.selected_predicates[unit] = []   
            else:
                raise ValueError(f"Column {unit} already selected")
            if unit not in self.current_predicates_column:
                self.current_predicates_column.append(unit)
            if unit in self.global_predicates and len(self.global_predicates[unit]) > 0:
                self.global_predicates[unit].append(str(LogicalOperator.AND))
            if unit not in self.vector_state['predicates']:
                self.vector_state['predicates'][unit] = []
        elif mode == 'operator':
            col_name = self.current_predicates_column[-1]
            self.selected_predicates[col_name].append(unit)
            if col_name in self.global_predicates:
                self.global_predicates[col_name].append(unit)
            self.vector_state['predicates'][col_name].append(OPERATORDICT[unit])
            if unit == str(Operator.IS_NOT_NULL) or unit == str(Operator.IS_NULL):
                self.vector_state['predicates'][col_name].append([0])
        elif mode == 'value':
            col_name = self.current_predicates_column[-1]
            if not isinstance(self.selected_predicates[col_name][-1], list):
                self.selected_predicates[col_name].append([unit])
                if col_name in self.global_predicates:
                    self.global_predicates[col_name].append([unit])
                self.vector_state['predicates'][col_name].append([ori_value + 1])
            else:
                self.selected_predicates[col_name][-1].append(unit)
                if col_name in self.global_predicates:
                    self.global_predicates[col_name][-1].append(unit)
                self.vector_state['predicates'][col_name][-1].append(ori_value + 1)

        elif mode == 'cond':
            col_name = self.current_predicates_column[-1]
            self.selected_predicates[col_name].append(unit)
            if col_name in self.global_predicates:
                self.global_predicates[col_name].append(unit)
            self.vector_state['predicates'][col_name].append(LOGICALOPERATORDICT[unit])
        if len(self.vector_state['predicates'][self.current_predicates_column[-1]]) > (2 + self.max_value_one_predicate) * self.max_predicates_per_col - 1:
            # print(self.vector_state['predicates'][self.current_predicates_column[-1]])
            raise ValueError(f"Predicate length for column {self.current_predicates_column[-1]} exceeds ")
    def update_projection(self, unit, mode):
        if mode == 'column':
            self.selected_projection.append([unit])
            if unit not in self.current_projection_column:
                self.current_projection_column.append(unit)
            if unit not in self.vector_state['projection']:
                self.vector_state['projection'][unit] = []
        else:
            self.selected_projection[-1].append(unit)
            column = self.current_projection_column[-1]
            self.vector_state['projection'][column].append(AGGREGATORDICT[unit])

    def update_filed(self, colname, value_idx, value, operator, cond):
        max_edge = self.numeric_predicate_fild[colname]['cluster'].default_interval[1]
        min_true = self.numeric_predicate_fild[colname]['true'].default_interval[0]
        max_true = self.numeric_predicate_fild[colname]['true'].default_interval[1]
        if cond == str(LogicalOperator.AND) or cond == None:
            if operator in [str(Operator.GT),str(Operator.GEQ)]:
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] & Domain(intervals=[(value_idx[0], max_edge + 1)], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] & Domain(intervals=[(value[0], max_true)], default_interval=(min_true, max_true))
            elif operator in [str(Operator.LT),str(Operator.LEQ)]:
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] & Domain(intervals=[(-1, value_idx[0])], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] & Domain(intervals=[(min_true, value[0])], default_interval=(min_true, max_true))
            elif operator == str(Operator.BETWEEN):
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] & Domain(intervals=[(min(value_idx), max(value_idx))], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] & Domain(intervals=[(min(value), max(value))], default_interval=(min_true, max_true))
            elif operator == str(Operator.EQ):
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] & Domain(intervals=[(value_idx[0], value_idx[0])], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] & Domain(intervals=[(value[0], value[0])], default_interval=(min_true, max_true))
        elif cond == str(LogicalOperator.OR):
            if operator in [str(Operator.GT),str(Operator.GEQ)]:
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] | Domain(intervals=[(value_idx[0], max_edge + 1)], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] | Domain(intervals=[(value[0], max_true)], default_interval=(min_true, max_true))
            elif operator in [str(Operator.LT),str(Operator.LEQ)]:
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] | Domain(intervals=[(-1, value_idx[0])], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] | Domain(intervals=[(min_true, value[0])], default_interval=(min_true, max_true))
            elif operator == str(Operator.BETWEEN):
                self.numeric_predicate_fild[colname]['cluster'] = self.numeric_predicate_fild[colname]['cluster'] | Domain(intervals=[(min(value_idx), max(value_idx))], default_interval=(0, max_edge))
                self.numeric_predicate_fild[colname]['true']    = self.numeric_predicate_fild[colname]['true'] | Domain(intervals=[(min(value), max(value))], default_interval=(min_true, max_true))
            # elif operator == str(Operator.NEQ)
        if self.numeric_predicate_fild[colname]['cluster'].min_value() > self.numeric_predicate_fild[colname]['cluster'].max_value():
            raise ValueError(f"Invalid predicate range for column {colname}: {self.numeric_predicate_fild[colname]}")
        
    def delete_possible_joins(self, joins):
        self.current_possible_joins[joins[0]].remove(joins[1])
        self.current_possible_joins[joins[1]].remove(joins[0])
        if len(self.current_possible_joins[joins[0]]) == 0:
            del self.current_possible_joins[joins[0]]
        if len(self.current_possible_joins[joins[1]]) == 0:
            del self.current_possible_joins[joins[1]]
            
    def update_global_predicates_keys(self, joins):
        if joins[0] not in self.global_predicates and joins[1] not in self.global_predicates:
            union_predicates = []
            self.global_predicates[joins[0]] = union_predicates
            self.global_predicates[joins[1]] = union_predicates
        elif joins[0] not in self.global_predicates:
            self.global_predicates[joins[0]] = self.global_predicates[joins[1]]
        elif joins[1] not in self.global_predicates:
            self.global_predicates[joins[1]] = self.global_predicates[joins[0]]
        
    def get_available_joins(self):
        available_join_columns = []
        if len(self.selected_joins) == 0:
            for left_col, right_cols in self.current_possible_joins.items():
                available_join_columns.append(self._possible_column[left_col])
        else:
            if len(self.selected_joins[-1]) == 2:
                for left_column in self.current_possible_joins:
                    table = left_column.split('.')[0]
                    if table in self.selected_tables:
                        available_join_columns.append(self._possible_column[left_column])
            elif len(self.selected_joins[-1]) == 1:
                for right_col in self.current_possible_joins[self.selected_joins[-1][0]]:
                    available_join_columns.append(self._possible_column[right_col])

        return available_join_columns
    def get_attn_matrix(self):

        max_column = AgentActionType.COL_END.value - AgentActionType.COL_START.value
        join_matrix = np.full((max_column, max_column), -1e9)
        if len(self.selected_joins) == 0:
            for left_col, right_cols in self.current_possible_joins.items():
                for right_col in right_cols:
                    join_matrix[self._possible_column[left_col], self._possible_column[right_col]] = 1e-2
                    join_matrix[self._possible_column[right_col], self._possible_column[left_col]] = 1e-2
        else:
            if len(self.selected_joins[-1]) == 2:
                for left_column in self.current_possible_joins:
                    table = left_column.split('.')[0]
                    if table in self.selected_tables:
                        for right_col in self.current_possible_joins[left_column]:
                            join_matrix[self._possible_column[left_column], self._possible_column[right_col]] = 1e-2
                            join_matrix[self._possible_column[right_col], self._possible_column[left_column]] = 1e-2
            elif len(self.selected_joins[-1]) == 1:
                for right_col in self.current_possible_joins[self.selected_joins[-1][0]]:
                    join_matrix[self._possible_column[self.selected_joins[-1][0]], self._possible_column[right_col]] = 1e-2
                    join_matrix[self._possible_column[right_col], self._possible_column[self.selected_joins[-1][0]]] = 1e-2
            for join in self.selected_joins:
                if len(join) == 2:
                    table1 = join[0].split('.')[0]
                    table2 = join[1].split('.')[0]
                    for col1_idx in self.table_to_cols[table1]:
                        for col2_idx in self.table_to_cols[table2]:
                            join_matrix[col1_idx, col2_idx] = 1e-3
                            join_matrix[col2_idx, col1_idx] = 1e-3
                    join_matrix[self._possible_column[join[0]], self._possible_column[join[1]]] = 1e-1
                    join_matrix[self._possible_column[join[1]], self._possible_column[join[0]]] = 1e-1                    

        for i in range(len(self._possible_column)):
            join_matrix[i,i] = 1e-1
        attn_bias = np.ones((max_column + 1, max_column + 1))
        attn_bias[1:, 1:] = join_matrix
        return attn_bias
    
    def get_available_predicates_columns(self):
        available_predicate_columns = []
        
        for table in self.selected_tables:
            for col in self._possible_column:
                if col.startswith(f"{table}.") and col not in self.current_predicates_column:
                    available_predicate_columns.append(self._possible_column[col])
        
        return available_predicate_columns
    
    def get_available_predicates_type(self, column):
        return self._possible_predicates_type[column]
    
    def get_available_projections_columns(self):
        available_projection_columns = []
    
        for table in self.selected_tables:
            for col in self._possible_aggregations:
                if col.startswith(f"{table}.") and col not in self.current_projection_column:
                    available_projection_columns.append(self._possible_column[col])
        
        return available_projection_columns
    
    def get_available_group_bys_columns(self):
        available_group_by_columns = []
    
        for table in self.selected_tables:
            for col in self._possible_column:
                if col.startswith(f"{table}."):
                    available_group_by_columns.append(self._possible_column[col])
        
        return available_group_by_columns

    def get_state_vector(self, action_type, action_mask):
        node_vector_dict = copy.deepcopy(self.baseline_state)

        # Define start indices based on calculated vector length components
        predicate_section_start_idx = 2
        # Size calculation based on len_node_vector definition in __init__
        predicate_section_size = (2 + self.max_value_one_predicate) * self.max_predicates_per_col - 1
        projection_section_start_idx = predicate_section_start_idx + predicate_section_size
        aggregation_section_size = self.max_aggregates_per_col # From len_node_vector definition
        # print(predicate_section_size,projection_section_start_idx,aggregation_section_size)
        # Process Predicates for each column
        for column, predicates in self.vector_state['predicates'].items():
            # Use a dedicated index for the current position within this column's vector
            idx_within_col = predicate_section_start_idx
            # Define the boundary for the predicate section for this column
            max_pred_idx = projection_section_start_idx

            for unit in predicates:
                # Stop if we've filled the predicate section for this column
                if idx_within_col >= max_pred_idx:
                    print(f"Warning: Exceeded predicate vector space for column {column}. Predicate part might be truncated.")
                    break

                if not isinstance(unit, list): # Operator or LogicalOperator code
                    node_vector_dict[column][idx_within_col] = unit
                    idx_within_col += 1 # Increment index after writing operator
                else: # Value list
                    # Write up to max_value_one_predicate values/padding, respecting boundaries
                    num_values_to_write = min(self.max_value_one_predicate, max_pred_idx - idx_within_col)
                    for i in range(num_values_to_write):
                        if i < len(unit):
                            node_vector_dict[column][idx_within_col] = unit[i] / (AgentActionType.VALUE_END.value - AgentActionType.VALUE_START.value)
                        else:
                            node_vector_dict[column][idx_within_col] = 0 # Padding
                        idx_within_col += 1 # Increment index for each value/padding slot

        # Process Projections for each column
        for column, projection in self.vector_state['projection'].items():
            # Start writing projections after the predicate section
            idx_within_col = projection_section_start_idx
            # Define the boundary for the projection section
            max_proj_idx = projection_section_start_idx + aggregation_section_size
            if len(projection) == 0:
                node_vector_dict[column][idx_within_col] = AGGREGATORDICT[str(Aggregator.NONE)]
                idx_within_col += 1
            else:
                for unit in projection: # Aggregator codes
                    # Stop if we've filled the projection section for this column
                    if idx_within_col >= max_proj_idx:
                        print(f"Warning: Exceeded aggregation vector space for column {column}. Aggregation part might be truncated.")
                        break

                    node_vector_dict[column][idx_within_col] = unit
                    idx_within_col += 1
        attn_bias = self.get_attn_matrix()

        # pad 
        node_vector = []
        for column in node_vector_dict:
            node_vector.append(node_vector_dict[column])
        for _ in range(len(node_vector), AgentActionType.COL_END.value - AgentActionType.COL_START.value):
            node_vector.append(np.zeros(len(node_vector[0])))
        # print(len(action_mask))
        state ={'node_vector':np.array(node_vector,dtype=np.int16),'attn_bias':attn_bias,
                'action_mask':np.array(action_mask,dtype=np.int16),'action_type':np.array([action_type],dtype=np.int16)}
        return state
    
    def to_query(self):
        if not self.is_complete:
            return "Query is not complete"
        def quote_column(col):
            table, col_name = col.split('.')
            return f'"{table}"."{col_name}"'
        select_clause = []
        if len(self.selected_projection) == 0:
            select_clause = ["COUNT(*)"]
        else:
            for proj in self.selected_projection:
                if len(proj) == 2:  # Has function
                    col_name = proj[0]
                    agg_func = proj[1]
                    if agg_func == str(Aggregator.COUNTDISTINCT):
                        select_clause.append(f'COUNT(DISTINCT {quote_column(col_name)})')
                    else:
                        select_clause.append(f'{agg_func}({quote_column(col_name)})')
                else:
                    select_clause.append(quote_column(proj[0]))
                
        
        select_str = "SELECT " + ", ".join(select_clause)
        
        # FROM clause
        from_str = "FROM " + ", ".join([f'"{t}"' for t in self.selected_tables])
        
        # WHERE clause (including join conditions)
        where_conditions = []
        
        # Add join conditions
        for join in self.selected_joins:
            if len(join) == 2:  # Complete join with left and right columns
                left_col = join[0]
                right_col = join[1]
                where_conditions.append(f'{quote_column(left_col)} = {quote_column(right_col)}')
        # Add predicates
        for colname, preds in self.selected_predicates.items():
            operator = None
            values = None
            one_predicates_str = ''
            count_predicates = 0
            for pred in preds:
                if isinstance(pred, str) and (pred == str(LogicalOperator.AND) or pred == str(LogicalOperator.OR)):
                    # Logical operator)'

                    count_predicates += 1
                    if values is not None:
                        one_predicates_str += ' '.join([quote_column(colname), operator, values])
                    else:
                        one_predicates_str += ' '.join([quote_column(colname), operator])
                    if count_predicates >= 2:
                        one_predicates_str = '(' + one_predicates_str + ')'
                    one_predicates_str += f' {pred} '
                    operator = None
                    values = None
                elif isinstance(pred, list):  # Complete predicate with column, operator, and value(s)
                    if operator == str(Operator.IN) or operator == str(Operator.NOT_IN):
                        values = '(' + ", ".join([str(v) for v in pred]) + ')'
                    elif operator == str(Operator.BETWEEN):
                        values = str(min(pred[0],pred[1])) + ' AND ' + str(max(pred[0],pred[1]))
                    else:
                        values = str(pred[0])
                else:
                    operator = pred
            if values is not None:
                    one_predicates_str += ' '.join([quote_column(colname), operator, values])
            else:
                one_predicates_str += ' '.join([quote_column(colname), operator])
            if count_predicates >= 1:
                one_predicates_str = '(' + one_predicates_str + ')'
            where_conditions.append(one_predicates_str)
        where_str = ""
        if where_conditions:
            where_str = "WHERE " + " AND ".join(where_conditions)
        
        # GROUP BY clause
        group_by_str = ""
        if self.selected_group_bys:
            group_by_cols = [col for col in self.selected_group_bys]
            group_by_str = "GROUP BY " + ", ".join(group_by_cols)
        
        # HAVING clause
        having_str = ""
        if self.having_clause:
            having_str = "HAVING " + self.having_clause
        
        # LIMIT clause
        limit_str = ""
        if self.limit:
            limit_str = f"LIMIT {self.limit}"
        
        # Assemble the query
        query_parts = [select_str, from_str]
        
        if where_str:
            query_parts.append(where_str)
        if group_by_str:
            query_parts.append(group_by_str)
        if having_str:
            query_parts.append(having_str)
        if limit_str:
            query_parts.append(limit_str)
        
        return " ".join(query_parts) + ";"

class SQLGenEnv:
    def __init__(self):
        super(SQLGenEnv, self).__init__()
        self.datasets = config.databases
        self.max_no_joins = 16
        self.max_no_predicates = 6
        self.max_no_aggregates = 3
        self.max_no_group_by = 3
        self.max_predicate_per_col = 3
        self.max_value_one_predicate = 10
        self.max_aggregates_per_col = 1
        self.max_group_by_per_col = 3
        self.group_by_threshold = 10000

        # self.pghelper = pghelper
        self.index_info = load_index_info()
        self.column_stats_dict = {}
        self.state_dict = {}
        for dataset in self.datasets:
            self.column_stats_dict[dataset] = load_column_statistics(dataset)
            schema = load_schema_json(dataset)
            

            possible_joins = collections.defaultdict(list)
            possible_predicates_type = {}
            possible_aggregations = []
            possible_group_by_columns = []

            for table_l, column_l, table_r, column_r in schema.relationships:
                if isinstance(column_l, list) and isinstance(column_r, list):
                    for i in range(len(column_l)):
                        possible_joins[table_l + '.' + column_l[i]].append(table_r + '.' + column_r[i])
                        possible_joins[table_r + '.' + column_r[i]].append(table_l + '.' + column_l[i])
                else:
                    possible_joins[table_l + '.' + column_l].append(table_r + '.' + column_r)
                    possible_joins[table_r + '.' + column_r].append(table_l + '.' + column_l)
            all_tables = vars(self.column_stats_dict[dataset])
            for table in all_tables:
                all_columns = vars(all_tables[table])
                for col in all_columns:
                    if all_columns[col].nan_ratio < 1 - 1e-5 and all_columns[col].num_unique > 1:
                        if all_columns[col].datatype == str(Datatype.INT) or all_columns[col].datatype == str(Datatype.FLOAT):
                            possible_predicates_type[table + '.' + col]=[AgentActionType.OP_GREATER.value, AgentActionType.OP_LESS.value, 
                                                                        AgentActionType.OP_GREATER_EQUAL.value,AgentActionType.OP_NOT_EQUAL.value, AgentActionType.OP_LESS_EQUAL.value,
                                                                            AgentActionType.OP_BETWEEN.value]
                            # in for INT?
                            possible_aggregations.append(table + '.' + col)
                        elif all_columns[col].datatype == str(Datatype.CATEGORICAL) or all_columns[col].datatype == str(Datatype.MISC):
                            possible_predicates_type[table + '.' + col] = []
                            if len(all_columns[col].sample_data) > 3:
                                possible_predicates_type[table + '.' + col].extend([AgentActionType.OP_NOT_EQUAL.value, AgentActionType.OP_NOT_IN.value])
                            if len(all_columns[col].freq_words) > 0:
                                possible_predicates_type[table + '.' + col].extend([AgentActionType.OP_LIKE.value, AgentActionType.OP_NOT_LIKE.value])
                            if all_columns[col].num_unique > 3 and all_columns[col].n_distinct < 0.5:
                                possible_predicates_type[table + '.' + col].append(AgentActionType.OP_IN.value)
                        # elif all_columns[col].datatype == str(Datatype.MISC):
                        #     possible_predicates_type[table + '.' + col] = [AgentActionType.OP_NOT_EQUAL.value]
                        #     if len(all_columns[col].freq_words) > 0:
                        #         possible_predicates_type[table + '.' + col].extend([AgentActionType.OP_LIKE.value, AgentActionType.OP_NOT_LIKE.value])
                        #     if all_columns[col].num_unique < 100 and all_columns[col].n_distinct < 0.5:
                        #         possible_predicates_type[table + '.' + col].append(AgentActionType.OP_IN.value)
                        elif all_columns[col].datatype == str(Datatype.NULL):
                            pass
                        else:
                            possible_predicates_type[table + '.' + col] = []
                            # raise NotImplementedError(f'Column type {all_columns[col].datatype} not implemented.')
                        if all_columns[col].nan_ratio > 0.2:
                            possible_predicates_type[table + '.' + col].append(AgentActionType.OP_IS_NULL.value)
                        if all_columns[col].nan_ratio < 0.8:
                            possible_predicates_type[table + '.' + col].append(AgentActionType.OP_IS_NOT_NULL.value)
                        if all_columns[col].num_unique < 100 and all_columns[col].n_distinct < 0.5: 
                            possible_predicates_type[table + '.' + col].append(AgentActionType.OP_EQUAL.value)
            
                        if all_columns[col].datatype in {str(d) for d in [Datatype.INT, Datatype.CATEGORICAL]} \
                            and all_columns[col].num_unique < self.group_by_threshold:
                            possible_group_by_columns.append([table + '.' + col])
            
            possible_column = {}
            for col in possible_predicates_type:
                possible_column[col] = len(possible_column)
            self.state_dict[dataset] = QueryState(dataset, self.index_info, possible_joins, possible_column, 
                                    possible_predicates_type, possible_aggregations, possible_group_by_columns,
                                    self.max_predicate_per_col, self.max_aggregates_per_col, 
                                    self.max_group_by_per_col, self.max_value_one_predicate)
        # self.action_space = spaces.Discrete(AgentActionType.COL_END + 1)
        # self.observation_space = spaces.Dict(
        #     {
        #         'node_vector': spaces.Box(low=float('-inf'), high=float('inf'), shape=(AgentActionType.COL_END - AgentActionType.COL_START, self.state_dict[self.datasets[0]].len_node_vector), dtype=np.int16),
        #         'attn_bias': spaces.Box(low=float('-inf'), high=float('inf'), shape=(AgentActionType.COL_END - AgentActionType.COL_START + 1, AgentActionType.COL_END - AgentActionType.COL_START + 1), dtype=np.float32),
        #         'action_mask': spaces.Box(low=0, high=1, shape=(AgentActionType.COL_END + 1,), dtype=np.int16),
        #         'action_type': spaces.Box(low=0, high=config.trigger_action_num, shape=(1,), dtype=np.int16)
        #     }
        # )

    def reset(self, options=None):
        self.column_stats = self.column_stats_dict[options['dbName']]
        self.state : QueryState = self.state_dict[options['dbName']]
        self.state.reset()
        self.last_step_type = None
        self.next_step_type = TriggerActionType.SELECT_LEFT_JOIN_COLUMN

        action_mask = self._get_next_step_action_space(self.next_step_type)
        state_vector = self.state.get_state_vector(self.next_step_type.value, action_mask)
        info = {}
        return state_vector, info
    
    def step(self, action):  # Join -> Filter -> Select -> From
        reward = 0
        done = False
        
        if not AgentActionType.is_value(action):
            self.last_step_type = self.next_step_type
            
        if action == AgentActionType.END_JOIN:
            self.next_step_type = TriggerActionType.SELECT_PREDICATE_COLUMN
            
        elif action == AgentActionType.END_PREDICATE:
            self.next_step_type = TriggerActionType.SELET_PROJECTION_COLUMN
            
        elif action == AgentActionType.END_PROJECTION:
            self.state.is_complete = True
            done = True
            reward = self._calculate_reward() 
                
        elif AgentActionType.is_binop(action):
            self.last_step_type = TriggerActionType.SELECT_BINOP_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE
            self.state.update_predicates(str(self._action_to_operator(action)), 'operator')
                
        elif action == AgentActionType.OP_IN:
            self.last_step_type = TriggerActionType.SELECT_IN_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE

            self.state.update_predicates(str(Operator.IN),'operator')
        elif action == AgentActionType.OP_NOT_IN:
            self.last_step_type = TriggerActionType.SELECT_NOT_IN_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE

            self.state.update_predicates(str(Operator.NOT_IN),'operator')
                
        elif action == AgentActionType.END_IN_OP:
            self.next_step_type = TriggerActionType.AFTER_ONE_PREDICATE
            
        elif action == AgentActionType.OP_LIKE:
            self.last_step_type = TriggerActionType.SELECT_LIKE_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE
            self.state.update_predicates(str(Operator.LIKE),'operator')
        elif action == AgentActionType.OP_NOT_LIKE:
            self.last_step_type = TriggerActionType.SELECT_NOT_LIKE_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE
            self.state.update_predicates(str(Operator.NOT_LIKE),'operator')
            
        elif action == AgentActionType.OP_IS_NULL:
            self.next_step_type = TriggerActionType.AFTER_ONE_PREDICATE
            self.state.update_predicates(str(Operator.IS_NULL),'operator')
            
        elif action == AgentActionType.OP_IS_NOT_NULL:
            self.next_step_type = TriggerActionType.AFTER_ONE_PREDICATE
            self.state.update_predicates(str(Operator.IS_NOT_NULL),'operator')
            
        elif action == AgentActionType.OP_BETWEEN:
            self.last_step_type = TriggerActionType.SELECT_BETWEEN_OPERATOR
            self.next_step_type = TriggerActionType.SELECT_VALUE

            self.state.update_predicates(str(Operator.BETWEEN),'operator')
            
        elif action == AgentActionType.COND_AND:
            self.next_step_type = TriggerActionType.SELECT_OPERATOR
            self.state.update_predicates(str(LogicalOperator.AND), 'cond')
            
        elif action == AgentActionType.COND_OR:
            self.next_step_type = TriggerActionType.SELECT_OPERATOR
            self.state.update_predicates(str(LogicalOperator.OR), 'cond')

        elif AgentActionType.is_aggfunction(action):
            self.next_step_type = TriggerActionType.AFTER_ONE_PROJECTION
            self.state.update_projection(str(self._action_to_operator(action)), 'function')

        elif AgentActionType.is_value(action):
            if self.last_step_type in [TriggerActionType.SELECT_BINOP_OPERATOR, TriggerActionType.SELECT_LIKE_OPERATOR, TriggerActionType.SELECT_NOT_LIKE_OPERATOR]:   
                selected_value = self._sample_value_for_current_predicate(action)
                self.next_step_type = TriggerActionType.AFTER_ONE_PREDICATE
                self.state.update_predicates(selected_value,'value',action - AgentActionType.VALUE_START)

            elif self.last_step_type == TriggerActionType.SELECT_BETWEEN_OPERATOR:
                if not isinstance(self.state.selected_predicates[self.state.current_predicates_column[-1]][-1], list): 
                    self.next_step_type = TriggerActionType.SELECT_VALUE
                else: 
                    self.next_step_type = TriggerActionType.AFTER_ONE_PREDICATE
                selected_value = self._sample_value_for_current_predicate(action)
                self.state.update_predicates(selected_value,'value',action - AgentActionType.VALUE_START)
    
            elif self.last_step_type == TriggerActionType.SELECT_IN_OPERATOR or self.last_step_type == TriggerActionType.SELECT_NOT_IN_OPERATOR:
                col_name = self.state.current_predicates_column[-1]
                if not isinstance(self.state.selected_predicates[col_name][-1], list): 
                    selected_value = self._sample_value_for_current_predicate(action)
                    self.next_step_type = TriggerActionType.SELECT_VALUE
                else: 
                    selected_value = self._sample_value_for_current_predicate(action)
                    attmpt = 0
                    value_list = self.state.selected_predicates[col_name][-1]
                    while attmpt < 3:
                        if selected_value in value_list:
                            selected_value = self._sample_value_for_current_predicate(action)
                            attmpt += 1
                        else:
                            break
                    self.next_step_type = TriggerActionType.SELECT_VALUE_WITH_IN_OPERATOR
                self.state.update_predicates(selected_value,'value',action - AgentActionType.VALUE_START)
                
            else:
                raise NotImplementedError(f'Action type {action} not implemented.')
            
        elif AgentActionType.is_column(action):
            col_idx = action - AgentActionType.COL_START
            col_name = list(self.state._possible_column.keys())[col_idx]
            if self.last_step_type == TriggerActionType.SELECT_LEFT_JOIN_COLUMN or self.last_step_type == TriggerActionType.AFTER_ONE_JOIN:
                self.next_step_type = TriggerActionType.SELECT_RIGHT_JOIN_COLUMN
                self.state.update_joins(col_name, 'left')
                reward += 0.1 
            elif self.last_step_type == TriggerActionType.SELECT_RIGHT_JOIN_COLUMN:
                self.next_step_type = TriggerActionType.AFTER_ONE_JOIN
                self.state.update_joins(col_name, 'right')            
            elif self.last_step_type == TriggerActionType.SELECT_PREDICATE_COLUMN or self.last_step_type == TriggerActionType.AFTER_ONE_PREDICATE:
                self.next_step_type = TriggerActionType.SELECT_OPERATOR
                self.state.update_predicates(col_name, 'column')
                
            elif self.last_step_type == TriggerActionType.SELET_PROJECTION_COLUMN or self.last_step_type == TriggerActionType.AFTER_ONE_PROJECTION:
                self.next_step_type = TriggerActionType.SELECT_AGG_FUNCTION
                self.state.update_projection(col_name, 'column')
            
            # elif self.last_step_type == TriggerActionType.SELECT_GROUP_BY_COLUMN:
            #   
            #     self.state.selected_group_bys.append([table, column, 0])  
                
            #     
            #     if len(self.state.selected_group_bys) < self.max_no_group_by and randstate.random() < 0.5:
            #         self.next_step_type = TriggerActionType.SELECT_GROUP_BY_COLUMN
            #     else:
            #         if randstate.random() < self.having_prob:
            #             self.next_step_type = TriggerActionType.SELECT_HAVING_COLUMN
            #         else:
            #             self.state.is_complete = True
            #             done = True
            #             reward = self._calculate_reward()
                
            # elif self.last_step_type == TriggerActionType.SELECT_HAVING_COLUMN:
            #     pass
                
            # elif self.last_step_type == TriggerActionType.AFTER_ONE_PREDICATE:
            #     self.next_step_type = TriggerActionType.SELECT_BINOP_OPERATOR
            #     self.state.update_predicates(col_name, 'column')
        else:
            raise NotImplementedError(f'Action type {action} not implemented.')
            
        action_mask = self._get_next_step_action_space(self.next_step_type)
        
        state_vector = self.state.get_state_vector(self.next_step_type.value, action_mask)
        return state_vector, reward, done, False,{}
    
    def _action_to_operator(self, action):
        action_to_op = {
            AgentActionType.OP_EQUAL: Operator.EQ,
            AgentActionType.OP_GREATER: Operator.GEQ,
            AgentActionType.OP_LESS: Operator.LEQ,
            AgentActionType.OP_GREATER_EQUAL: Operator.GT,
            AgentActionType.OP_LESS_EQUAL: Operator.LT,
            AgentActionType.OP_NOT_EQUAL: Operator.NEQ,
            AgentActionType.OP_LIKE: Operator.LIKE,
            AgentActionType.OP_NOT_LIKE: Operator.NOT_LIKE,
            AgentActionType.OP_IS_NULL: Operator.IS_NULL,
            AgentActionType.OP_IS_NOT_NULL: Operator.IS_NOT_NULL,
            AgentActionType.OP_BETWEEN: Operator.BETWEEN,
            AgentActionType.OP_IN: Operator.IN,
            AgentActionType.OP_NOT_IN: Operator.NOT_IN,
            AgentActionType.FUNCTION_AVG: Aggregator.AVG,
            AgentActionType.FUNCTION_SUM: Aggregator.SUM,
            AgentActionType.FUNCTION_COUNT: Aggregator.COUNT,
            AgentActionType.FUNCTION_COUNTDISTINCT: Aggregator.COUNTDISTINCT,
            AgentActionType.FUNCTION_MIN: Aggregator.MIN,
            AgentActionType.FUNCTION_MAX: Aggregator.MAX,
            # AgentActionType.FUNCTION_NONE: Aggregator.NONE
        }
        if isinstance(action, np.ndarray):
            action = action.item()
        return action_to_op.get(action)

    def _sample_value_for_current_predicate(self, action):
        percentiles = action - AgentActionType.VALUE_START
        col_name = self.state.current_predicates_column[-1]
        current_pred = self.state.selected_predicates[col_name]
        table, column = col_name.split('.')
        
        col_stats = vars(vars(self.column_stats)[table]).get(column)
        operator = current_pred[-1]
        # cond = current_pred[-2] if len(current_pred) > 1 else None
        if col_stats is None:
            return 0
        if operator in [str(Operator.LIKE), str(Operator.NOT_LIKE)]:
            sample_data = col_stats.freq_words
            if isinstance(sample_data[percentiles], list):
                value = rand_choice(randstate, sample_data[percentiles])
            else:
                value = sample_data[percentiles]
            value = value.replace("'", "''")
            return f"'{value}'"
        else:
            all_sample_data = col_stats.sample_data
            
            if isinstance(all_sample_data[percentiles], list):
                if col_stats.datatype == str(Datatype.INT) or col_stats.datatype == str(Datatype.FLOAT):
                    min_true = col_stats.min
                    max_true = col_stats.max
                    value_range = Domain(intervals=[(all_sample_data[percentiles][0], all_sample_data[percentiles][-1])],default_interval=(min_true,max_true))
                    value_range = value_range & self.state.numeric_predicate_fild[col_name]['true']
                    if value_range:
                        if value_range.min_value() == value_range.max_value():
                            value = value_range.min_value()
                        else:
                            sample_data = [s for s in all_sample_data[percentiles] if s >= value_range.min_value() and s <= value_range.max_value()]
                            value = rand_choice(randstate, sample_data)
                    else:
                        sample_data = all_sample_data[percentiles]
                        value = rand_choice(randstate, sample_data)
                else:
                    value = rand_choice(randstate, all_sample_data[percentiles])
            else:
                value = all_sample_data[percentiles]
            if col_stats.datatype == str(Datatype.INT):
                return int(value)
            elif col_stats.datatype == str(Datatype.FLOAT):
                return float(value)
            else:
                value = value.replace("'", "''")
                return f"'{value}'"

    def _calculate_reward(self):
        # reward = self.pghelper.get_result(hint='', sql=self.state.to_query(), dbName=self.state.dataset)
        # reward = 0
        # planJson = self.pghelper.get_cost_plan(hint='', sql=self.state.to_query(), hintStyle=0, dbName=self.state.dataset)
        # reward = math.log10(1 + int(planJson['Plan']['Total Cost']))

        return 0
        
    def _check_value_range(self, col_name, col_stats, last_logical_op, operator):
        max_edge = len(col_stats.sample_data) - 1
        if col_name not in self.state.numeric_predicate_fild:
            self.state.numeric_predicate_fild[col_name] = {'true':Domain(intervals=[(col_stats.min, col_stats.max)], default_interval=(col_stats.min, col_stats.max)),
                                                           'cluster':Domain(intervals=[(-1, max_edge + 1)], default_interval=(0, max_edge))} # [0,len(col_stats.sample_data) - 1]
        current_range = self.state.numeric_predicate_fild[col_name]['cluster']
        available_range = None
        if last_logical_op == str(LogicalOperator.OR):
            if current_range.min_value() <= -1 and current_range.max_value() >= max_edge + 1:
                available_range = current_range.get_in_range()
            else:
                if operator == str(Operator.GT) or operator == str(Operator.GEQ):
                    # available_range = current_range.get_not_in_range()
                    available_range = Domain(intervals=[(current_range.max_value(), max_edge + 1)], default_interval=(0, max_edge)).get_in_range()
                    # available_range.append([current_range[1],len(col_stats.sample_data) - 1])  #
                elif operator == str(Operator.LT) or operator == str(Operator.LEQ):
                    available_range = Domain(intervals=[(-1, current_range.min_value())], default_interval=(0, max_edge)).get_in_range()
                    # available_range = [0,current_range[0]]
                elif operator == str(Operator.BETWEEN) or operator == str(Operator.EQ):
                    
                    if current_range.min_value() == -1:
                        available_range = Domain(intervals=[(current_range.max_value(), max_edge + 1)], default_interval=(0, max_edge)).get_in_range()
                    elif current_range.max_value() == max_edge + 1:
                        available_range = Domain(intervals=[(-1, current_range.min_value())], default_interval=(0, max_edge)).get_in_range()
                    elif current_range.min_value() != -1 and current_range.max_value() != max_edge + 1:
                        available_range = Domain(intervals=[(current_range.max_value(), max_edge + 1), (-1, current_range.min_value())], 
                                                default_interval=(0, max_edge)).get_in_range()
                # if current_range[0] == -1:
                #     available_range.append([current_range[1],len(col_stats.sample_data) - 1])
                # elif current_range[1] == len(col_stats.sample_data):
                #     available_range.append([0,current_range[0]])
                # else:
                #     available_range.append([0,current_range[0]])
                #     available_range.append([current_range[1],len(col_stats.sample_data) - 1])
            
        elif last_logical_op == str(LogicalOperator.AND) or last_logical_op == None:
            # available_range.append([current_range[0],current_range[1]])
            available_range = current_range.get_in_range()
        # if available_range[0] == -1:
        #     available_range[0] = 0
        # if available_range[1] == len(col_stats.sample_data):
        #     available_range[1] = len(col_stats.sample_data) - 1
        return available_range

    def _get_next_step_action_space(self, action_type): # manage the high-level action space
        action_mask = np.zeros(AgentActionType.COL_END + 1, dtype=np.int8)
        
        if action_type == TriggerActionType.SELECT_LEFT_JOIN_COLUMN:
            available_columns = self.state.get_available_joins()
            for col_idx in available_columns:
                action_mask[col_idx + AgentActionType.COL_START] = 1
            
        elif action_type == TriggerActionType.SELECT_RIGHT_JOIN_COLUMN:

            available_columns = self.state.get_available_joins()
            for col_idx in available_columns:
                action_mask[col_idx + AgentActionType.COL_START] = 1
        
        elif action_type == TriggerActionType.SELECT_PREDICATE_COLUMN:
            # Select predicate columns from tables that don't have predicates or projections yet
            available_columns = self.state.get_available_predicates_columns()
            
            for col_idx in available_columns:
                action_mask[col_idx + AgentActionType.COL_START] = 1
        
        elif action_type == TriggerActionType.SELET_PROJECTION_COLUMN:
            available_columns = self.state.get_available_projections_columns()
            action_mask[AgentActionType.END_PROJECTION] = 1
            for col_idx in available_columns:
                action_mask[col_idx + AgentActionType.COL_START] = 1
        
        elif action_type == TriggerActionType.SELECT_OPERATOR:
            col_name = self.state.current_predicates_column[-1]
            if col_name in self.state.global_predicates:
                current_predicates = self.state.global_predicates[col_name]
            else:
                current_predicates = self.state.selected_predicates[col_name]
            available_operators = self.state.get_available_predicates_type(col_name)
            last_logical_op = current_predicates[-1] if len(current_predicates) > 1 else None 
            pruned_operators = []
            if col_name in self.state.numeric_predicate_fild:
                avaliable_range = self.state.numeric_predicate_fild[col_name]['cluster']
                max_edge = avaliable_range.default_interval[1] + 1
            else:
                avaliable_range = None
            has_in = False
            has_is_not_null = False
            has_is_null = False
            for unit in current_predicates:
                if isinstance(unit, str):
                    if unit == str(Operator.IN):
                        has_in = True
                    elif unit == str(Operator.IS_NULL):
                        has_is_null = True
                    elif unit == str(Operator.IS_NOT_NULL):
                        has_is_not_null = True
            for op_action_value in available_operators:
                op_str = str(self._action_to_operator(op_action_value))
                if op_str is None: continue

                allow_op = True # Assume allowed initially

                if last_logical_op == str(LogicalOperator.AND):
                    if op_str in [str(Operator.IS_NULL), str(Operator.IS_NOT_NULL), 
                                  str(Operator.EQ), str(Operator.IN), str(Operator.BETWEEN)]:
                        allow_op = False
                    if avaliable_range is not None:
                        if avaliable_range.min_value() == -1 and avaliable_range.max_value() == max_edge:
                            pass
                        elif avaliable_range.min_value() == -1 and op_str in [str(Operator.LEQ), str(Operator.LT)]:
                            allow_op = False
                        elif avaliable_range.max_value() == max_edge and op_str in [str(Operator.GEQ), str(Operator.GT)]:
                            allow_op = False
                        elif avaliable_range.min_value() != -1 and avaliable_range.max_value() != max_edge:
                            if op_str in [str(Operator.LEQ), str(Operator.LT), str(Operator.GEQ), str(Operator.GT)]:
                                allow_op = False
                elif last_logical_op == None:
                    pass
                elif last_logical_op == str(LogicalOperator.OR):
                    if op_str in [str(Operator.IS_NOT_NULL), str(Operator.NEQ), str(Operator.NOT_IN)]:
                        allow_op = False
                    if has_is_not_null or has_is_null:
                        if op_str in [str(Operator.IS_NULL), str(Operator.IS_NOT_NULL)]:
                            allow_op = False
                    if avaliable_range is not None:
                        if avaliable_range.min_value() == -1 and avaliable_range.max_value() == max_edge:
                            pass
                        elif avaliable_range.min_value() == -1 and op_str in [str(Operator.LEQ), str(Operator.LT)]:
                            allow_op = False
                        elif avaliable_range.max_value() == max_edge and op_str in [str(Operator.GEQ), str(Operator.GT)]:
                            allow_op = False
                if has_in:
                    if op_str == str(Operator.IN):
                        allow_op = False
                if allow_op:
                    pruned_operators.append(op_action_value)
            if len(pruned_operators) == 0:
                action_mask[AgentActionType.END_PREDICATE] = 1
                if col_name in self.state.global_predicates:
                    self.state.global_predicates[col_name].pop()
                else:
                    self.state.selected_predicates[col_name].pop()  
            else:
                for op in pruned_operators:
                    action_mask[op] = 1
            
        elif action_type == TriggerActionType.SELECT_VALUE or action_type == TriggerActionType.SELECT_VALUE_WITH_IN_OPERATOR:
            # For numeric columns, restrict the value range based on existing predicates
            col_name = self.state.current_predicates_column[-1]
            if col_name in self.state.global_predicates:
                current_predicates = self.state.global_predicates[col_name]
            else:
                current_predicates = self.state.selected_predicates[col_name]
            table, column = col_name.split('.')
            col_stats = vars(vars(self.column_stats)[table]).get(column)
            if isinstance(current_predicates[-1], list):     # modify
                selected_operator = current_predicates[-2] 
                cond = current_predicates[-3] if len(current_predicates) > 2 else None
            else:
                selected_operator = current_predicates[-1]
                cond = current_predicates[-2] if len(current_predicates) > 1 else None
            if selected_operator in [str(Operator.LIKE), str(Operator.NOT_LIKE)]:
                value_range = len(col_stats.freq_words)
                for action_idx in range(AgentActionType.VALUE_START, AgentActionType.VALUE_START + value_range):
                    action_mask[action_idx] = 1
            else:
                # For numeric columns, check value ranges
                if col_stats.datatype in [str(Datatype.INT), str(Datatype.FLOAT)]:
                    # Get valid value range
                    available_range = self._check_value_range(col_name, col_stats, cond, selected_operator)
                    if not available_range:
                        print(col_name, selected_operator, cond)
                    for avarange in available_range:
                        # for idx in range(avarange[0], avarange[1] + 1):
                        action_idx = AgentActionType.VALUE_START + avarange
                        if action_idx <= AgentActionType.VALUE_END:
                            action_mask[action_idx] = 1
                    if not any(action_mask[AgentActionType.VALUE_START:AgentActionType.VALUE_END + 1]):
                        print(col_name, selected_operator, cond)
                else:
                    value_range = len(col_stats.sample_data)
                    for action_idx in range(AgentActionType.VALUE_START, AgentActionType.VALUE_START + value_range):
                        if action_idx <= AgentActionType.VALUE_END:
                            action_mask[action_idx] = 1
                            
            if action_type == TriggerActionType.SELECT_VALUE_WITH_IN_OPERATOR:
                if len(current_predicates[-1]) + 1 >= self.max_value_one_predicate or len(current_predicates[-1]) > (len(col_stats.sample_data) - 1):
                    action_mask = np.zeros(AgentActionType.COL_END + 1, dtype=np.int8)
                action_mask[AgentActionType.END_IN_OP] = 1

        elif action_type == TriggerActionType.AFTER_ONE_PREDICATE:
            col_name = self.state.current_predicates_column[-1]
            if col_name in self.state.global_predicates:
                current_predicates = self.state.global_predicates[col_name]
            else:
                current_predicates = self.state.selected_predicates[col_name]
            col_stats = vars(vars(self.column_stats)[col_name.split('.')[0]]).get(col_name.split('.')[1])
            has_is_null = False
            has_is_not_null = False
            has_neq = False
            is_range = True
            expr = ''
            conds_counts = 0
            for unit in current_predicates:
                if isinstance(unit, str):
                    if unit in OPERATORDICT:
                        if unit == str(Operator.IS_NULL):
                            has_is_null = True
                        elif unit == str(Operator.IS_NOT_NULL):
                            has_is_not_null = True
                        elif unit == str(Operator.EQ) or unit == str(Operator.IN):
                            is_range = False
                        elif unit == str(Operator.NEQ) or unit == str(Operator.NOT_IN):
                            has_neq = True
                        expr += str(is_range) + ' '
                    elif unit in [str(LogicalOperator.AND), str(LogicalOperator.OR)]:
                        conds_counts += 1
                        expr += str(unit).lower() + ' '
            is_point = not eval(expr)
            if conds_counts < (self.max_predicate_per_col - 1):  
                if has_is_not_null and has_is_null:
                    pass
                elif has_is_not_null:
                    pass
                elif has_is_null:
                    action_mask[AgentActionType.COND_OR] = 1
                else:
                    action_mask[AgentActionType.COND_AND] = 1
                    action_mask[AgentActionType.COND_OR] = 1
                if is_point:
                    action_mask[AgentActionType.COND_AND] = 0
                if has_neq:
                    action_mask[AgentActionType.COND_OR] = 0
            if col_stats.datatype in [str(Datatype.INT), str(Datatype.FLOAT)]:
                value = current_predicates[-1]   
                if isinstance(value, list):   
                    
                    selected_operator = current_predicates[-2]
                    if selected_operator in [str(Operator.GT), str(Operator.GEQ),str(Operator.LT), str(Operator.LEQ), str(Operator.BETWEEN)]:
                        value_idx = []
                        cond = current_predicates[-3] if len(current_predicates) > 2 else None
                        for v in value:
                            val_idx = -1
                            for idx, samples in enumerate(col_stats.sample_data):
                                if isinstance(samples, list):
                                    edge_value = samples[0]
                                else:
                                    edge_value = samples
                                if v < edge_value:
                                    val_idx = (idx - 1)
                                    break
                            if val_idx == -1:
                                val_idx = len(col_stats.sample_data) - 1 
                            value_idx.append(val_idx)
                        self.state.update_filed(col_name, value_idx, value, selected_operator, cond)
            if len(self.state.selected_predicates) < self.max_no_predicates:
                available_columns = self.state.get_available_predicates_columns()
                for col_idx in available_columns:
                    action_mask[col_idx + AgentActionType.COL_START] = 1
            action_mask[AgentActionType.END_PREDICATE] = 1
        elif action_type == TriggerActionType.AFTER_ONE_JOIN:
     
            action_mask[AgentActionType.END_JOIN] = 1

            if len(self.state.selected_joins) < self.max_no_joins:
                available_columns = self.state.get_available_joins()
                for col_idx in available_columns:
                    action_mask[col_idx + AgentActionType.COL_START] = 1
        
        elif action_type == TriggerActionType.SELECT_AGG_FUNCTION:
            for agg in [AgentActionType.FUNCTION_MIN, AgentActionType.FUNCTION_MAX, AgentActionType.FUNCTION_AVG, 
                        AgentActionType.FUNCTION_SUM, AgentActionType.FUNCTION_COUNTDISTINCT, AgentActionType.FUNCTION_COUNT]:
                action_mask[agg] = 1
        elif action_type == TriggerActionType.AFTER_ONE_PROJECTION:
            action_mask[AgentActionType.END_PROJECTION] = 1
            if len(self.state.selected_projection) < self.max_no_aggregates:
                available_columns = self.state.get_available_projections_columns()
                for col_idx in available_columns:
                    action_mask[col_idx + AgentActionType.COL_START] = 1

        # elif action_type == TriggerActionType.SELECT_GROUP_BY_COLUMN:
        #     available_columns = self.state.get_available_group_by_columns()
        #     for col_idx in available_columns:
        #         action_mask[col_idx + AgentActionType.COL_START] = 1
        
        return action_mask

    def get_query(self):
        return self.state.to_query()

def sample_query_using_rl():
    from LQO.SQLBuilder import SQLBuilder
    from LQO.planhelper import PlanHelper
    from LQO.config import Config
    config = Config()
    plan_helper = PlanHelper(config,build_pghelper=True)
    sql_builder = SQLBuilder(plan_helper)
    env = SQLGenEnv()
    for i in range(10000):
        state,info = env.reset(options = {'dbName': 'chembl'})
        done = False
        
        while not done:
            valid_actions = [i for i, mask in enumerate(state['action_mask']) if mask == 1]
            action = randstate.choice(valid_actions) if valid_actions else 0
            state, reward, done, truncated, info = env.step(action)
        print(f"query {i}:")
        sql = env.get_query()
        sql_builder.get_unique_code(sql,'chembl')
    
    return sql
if __name__ == "__main__":
    
    sample_query_using_rl()