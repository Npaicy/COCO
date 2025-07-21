import random
from matplotlib import pyplot as plt
import numpy as np
from typing import List
import ray
import os
import torch
import math
# import matplotlib.pyplot as plt
from LQO.constant import PLANMAXTIMEOUT

class Node:
    def __init__(self, env_state, plan_state, action=None, parent=None):
        self.env_state = env_state
        self.plan_state = plan_state
        self.action = action  # Action that led to this state
        self.parent = parent
        self.children = {}  # action -> Node
        self.visited = False
        self.code = 0  # Track the current hint code

class Plan:
    def __init__(self, hint_code, plan_str):
        self.hint_code = hint_code
        self.plan_str = plan_str

    def __eq__(self, other):
        return self.plan_str == other.plan_str
    
    def __repr__(self):
        return self.plan_str
    
    def update(self, plan_json, latency, istimeout, feature_dict, hint_dict):
        self.plan_json = plan_json
        self.latency = latency
        self.istimeout = istimeout
        self.feature_dict = feature_dict
        self.hint_dict = hint_dict
class SQL:
    def __init__(self, dbName: str, sql_statement: str, plans: List[Plan]):
        self.dbName = dbName
        self.sql_statement = sql_statement
        # self.sql_buffer_reward = None
        self.pos_in_sqlgen_buffer = None
        self.sqlgen_reward = None
        self.plans = {}
        self.q_min_latency = None
        if plans:
            self.min_latency = plans[0].latency
            self.max_latency = plans[0].latency
            for plan in plans:
                self.plans[plan.hint_code] = plan
                self.min_latency = min(self.min_latency, plan.latency)
                self.max_latency = max(self.max_latency, plan.latency)
                # if (self.min_latency - plan.latency) / self.min_latency > 0.05:
                #     self.min_latency = plan.latency  
            self.base_latency = self.plans[0].latency
            self.base_reward = 0.0 # 3 * math.log2((self.base_latency - self.min_latency) + 1) / math.log2(PLANMAXTIMEOUT) #math.log10(self.base_latency - self.min_latency + 1.0) # base reward 越大，表示任务优化空间越大
        else:
            self.min_latency = None
            self.base_latency = None
        
    def get_plan_by_hint_code(self, hint_code: int):
        return self.plans[hint_code]
    
    def update_reward(self,  sqlgen_reward: float):
        # self.sql_buffer_reward = sql_buffer_reward
        self.sqlgen_reward = sqlgen_reward + self.base_reward

    def update_pos_in_sqlgen_buffer(self, pos: int):
        self.pos_in_sqlgen_buffer = pos

    def update_idx(self, idx: str):
        self.id = idx
            
class SQLBuffer:
    def __init__(self):
        self.buffer_idx = {}
        self.buffer = []
        self.capacity = 6000
        self.position = 0
        self.max_priority = 1.0
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.sample_prob = np.zeros((self.capacity,), dtype=np.float32)
        self.sampled_max_idx = 0
        self.epsilon = 1e-6
        self.update_times = 0
        self.one_update = False
        self.sample_strategy = 0

    def push(self, sql: SQL, pos: int):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if sql.dbName not in self.buffer_idx:
            self.buffer_idx[sql.dbName] = 0

        self.buffer_idx[sql.dbName] += 1
        sql_id = sql.dbName + '_' + str(self.buffer_idx[sql.dbName])
        sql.update_idx(sql_id)
        sql.update_pos_in_sqlgen_buffer(pos)
        self.buffer[self.position] = sql
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        self.sampled_max_idx = min(self.capacity, self.position)
        self.one_update = False

        return sql_id, pos
    
    def update_sample_stragety(self, strategy: str):
        if strategy == 'random':
            self.sample_strategy = 0
        elif strategy == 'hybrid':
            self.sample_strategy = 1
        elif strategy == 'priority':
            self.sample_strategy = 2
            
    def sample(self):
        if self.sample_strategy == 0:
            idx = np.random.randint(0, self.sampled_max_idx)
            return self.buffer[idx]
        elif self.sample_strategy == 1:
            if random.random() < 0.5: # hybrid sampling
                indices = np.random.choice(self.sampled_max_idx, 1, p=self.sample_prob)
                return self.buffer[indices[0]]
            else:
                idx = np.random.randint(0, self.sampled_max_idx)
                return self.buffer[idx]
        elif self.sample_strategy == 2:
            indices = np.random.choice(self.sampled_max_idx, 1, p=self.sample_prob)
            sql = self.buffer[indices[0]]
            return sql
        else:
            return None
                
    def update_priorities(self):
        self.position = len(self.buffer)
        for idx, sql in enumerate(self.buffer):
            self.priorities[idx] = np.clip(self.priorities[idx] + sql.sql_buffer_reward, 0.0, 10.0)
            # self.priorities[idx] = 1.0 
        self.max_priority = max(self.max_priority, np.max(self.priorities))

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.position]
        self.sample_prob = prios / prios.sum()
        self.sampled_max_idx = len(prios)

    def reset_priorities(self):
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        
    def paint_sample_prob(self):
        if len(self.buffer) > 0:
            # Create directory for plots if it doesn't exist
            os.makedirs("plots", exist_ok=True)
            
            # Create the plot
            plt.figure(figsize=(10, 6),dpi=300)
            plt.bar(range(len(self.sample_prob)), self.sample_prob)
            plt.xlabel('SQL Index')
            plt.ylabel('Sampling Probability')
            plt.title('SQL Buffer Sampling Probability Distribution')
            
            # Add some statistics as text
            stats_text = f"Max: {self.sample_prob.max():.4f}\n"
            stats_text += f"Min: {self.sample_prob.min():.4f}\n"
            stats_text += f"Mean: {self.sample_prob.mean():.4f}\n"
            stats_text += f"Buffer Size: {len(self.buffer)}"
            plt.figtext(0.02, 0.02, stats_text, fontsize=8)
            
            # Save the plot
            plt.savefig(f"plots/sample_prob_dist_{self.update_times}.png")
            plt.close()
            self.update_times += 1
            
    def get_sql_buffer(self):
        return self.buffer, self.one_update
    
    def get_buffer_size(self):
        return len(self.buffer)
    
    def update_sql_reward(self, idx: int, sqlgen_reward: float):
        self.buffer[idx].update_reward(sqlgen_reward)
        if idx == self.position - 1:
            self.one_update = True

    def save_state(self, checkpoint_dir):
        """Save buffer state to checkpoint directory"""
        if checkpoint_dir.endswith('.pkl'):
            buffer_path = checkpoint_dir
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            buffer_path = os.path.join(checkpoint_dir, "sql_buffer.pkl")
        buffer_state = {
            'buffer': self.buffer,
            'buffer_size': len(self.buffer),
            'position': self.position,
            'priorities': self.priorities,
            'buffer_idx': self.buffer_idx,
            'max_priority': self.max_priority,
            'sample_prob': self.sample_prob,
            'epsilon': self.epsilon
        }
        torch.save(buffer_state, buffer_path)
        return buffer_path

    def load_state(self, checkpoint_dir):
        """Load buffer state from checkpoint directory"""
        if checkpoint_dir.endswith('.pkl'):
            buffer_path = checkpoint_dir
        else:
            buffer_path = os.path.join(checkpoint_dir, "sql_buffer.pkl")
        if not os.path.exists(buffer_path):
            return False
        print(f"Loading buffer state from {buffer_path}")
        buffer_state = torch.load(buffer_path)
        self.buffer = buffer_state['buffer']
        self.position = buffer_state['position']
        self.priorities = buffer_state['priorities']
        self.buffer_idx = buffer_state['buffer_idx']
        self.max_priority = buffer_state['max_priority']
        self.sample_prob = buffer_state['sample_prob']
        self.epsilon = buffer_state.get('epsilon', self.epsilon)  # Use default if not found
        self.sampled_max_idx = self.position if self.position < self.capacity else self.capacity

        for sql in self.buffer:
            sql:SQL
            sql.update_pos_in_sqlgen_buffer(-1)
        # Recalculate sampling probabilities to ensure consistency
        # self.update_priorities()
        return True
    def update_buffer(self, buffer: List[SQL]):
        self.buffer = buffer
        self.position = len(self.buffer)

@ray.remote
class RaySQLBuffer:
    def __init__(self):
        self.sql_buffer = SQLBuffer()
    def push(self, sql: SQL, pos: int):
        return self.sql_buffer.push(sql, pos)
    def sample(self):
        return self.sql_buffer.sample()
    def update_priorities(self):
        return self.sql_buffer.update_priorities()
    def update_buffer(self, buffer: List[SQL]):
        return self.sql_buffer.update_buffer(buffer)
    def reset_priorities(self):
        return self.sql_buffer.reset_priorities()
    def get_sql_buffer(self):
        return self.sql_buffer.get_sql_buffer()
    def get_buffer_size(self):
        return self.sql_buffer.get_buffer_size()
    def update_sql_reward(self, idx: int, sql_buffer_reward: float, sqlgen_reward: float):
        return self.sql_buffer.update_sql_reward(idx, sql_buffer_reward, sqlgen_reward)
    def save_state(self, checkpoint_dir):
        return self.sql_buffer.save_state(checkpoint_dir)
    def load_state(self, checkpoint_dir):
        return self.sql_buffer.load_state(checkpoint_dir)
    
    def update_sample_stragety(self, strategy: str):
        return self.sql_buffer.update_sample_stragety(strategy)
