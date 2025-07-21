#  retrain LQO
from LQO.config import Config as LQOConfig
from SQLGen.config import Config as SQLGenConfig
from LQO.LQOEnv import  LQOEnvTest
from SQLGen.SQLGenEnv import SQLGenEnv
from SQLGen.SACAgent import SACAgent
from LQO.SQLBuffer import SQL,Plan, Node, SQLBuffer
from LQO.SQLBuilder import SQLBuilderActor
from LQO.LQOEnv import LQOEnvExp
# from LQO.IQLAgent import IQLAgent
from LQO.LQOAgent import LQOAgent, QDataset
from LQO.constant import PLANMAXTIMEOUT, MAXNODE, HINT2POS
import sys
import math
import pickle
import random
import os
import numpy as np
import torch

import ray
import logging
# # Set up root logger with console and file handlers
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)
# File handler
file_handler = logging.FileHandler('./train.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
# Module-specific logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class SQLGenTrainerActor:
    def __init__(self, sql_buffer:SQLBuffer, seed=1408, load_pretrain=False):
        set_seed(seed)
        self.sqlgen_config = SQLGenConfig()
        self.sqlgen_env = SQLGenEnv()
        self.sql_buffer:SQLBuffer = sql_buffer
        self.sqlgen_agent = SACAgent(**self.sqlgen_config.train_params)
        if load_pretrain and os.path.exists(self.sqlgen_config.pretrain_ckpt_dir):
            self.sqlgen_agent.load_models(self.sqlgen_config.pretrain_ckpt_dir, is_load_buffer=False, only_models=True)
        self.builder_actors = []
        for dbConfig in self.sqlgen_config.remote_db_config_list:
            config = LQOConfig()
            config.dbConfig = dbConfig
            self.builder_actors.append(SQLBuilderActor.remote(config))
        self.pending_state = None
        self.train_loops = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        os.makedirs(self.sqlgen_config.checkpoint_dir, exist_ok=True)
        self.test_databases = self.sqlgen_config.test_databases
        self.train_databases = [db for db in self.sqlgen_env.datasets if db not in self.sqlgen_config.test_databases]

    def iterate_one_loop(self, num_generated_sql: int):
        buffer_contents, one_update = self.sql_buffer.get_sql_buffer()
        self.logger.info(f"SQLGenTrainerActor: train_loops={self.train_loops}")
        self._update_sqlgen_reward(buffer_contents)
        self._update_sqlgen_model()
        self.logger.info(f"SQLGenTrainerActor: save_models")
        self.sqlgen_agent.save_models(self.sqlgen_config.sqlgen_agent_path)
        self.pending_state = self.generate_N_sql(num_generated_sql)
        self.train_loops += 1
        

    def generate_N_sql(self, num_generated_sql: int):
        pending = self.pending_state or {}
        backlog = []
        for _ in range(num_generated_sql):
            states = []
            for db in self.train_databases:
                state, _ = self.sqlgen_env.reset(options={'dbName': db})
                states.append(state)
            db_idx = self.sqlgen_agent.select_dbs(states)
            dbName = self.train_databases[db_idx]
            sql_statement, one_episode = self._generate_one_sql(dbName)
            backlog.append((dbName, sql_statement, one_episode))

        done, _ = ray.wait(list(pending.keys()), num_returns=len(pending), timeout=0.1)
        for fut in done:
            actor, dbName, one_episode = pending.pop(fut)
            self._try_collect_sql(fut, dbName, one_episode)

        busy = {actor for actor, _,  _ in pending.values()}
        available = [actor for actor in self.builder_actors if actor not in busy]
        for actor in available:
            if backlog:
                dbName, sql_statement, one_episode = backlog.pop(0)
                new_fut = actor.test_build_sql.remote(dbName, sql_statement)
                pending[new_fut] = (actor, dbName, one_episode)

        while backlog or len(pending) > (len(self.builder_actors) - 1):
            done, _ = ray.wait(list(pending.keys()), num_returns=1)
            fut = done[0]
            actor, dbName, one_episode = pending.pop(fut)
            self._try_collect_sql(fut, dbName, one_episode)

            if backlog:
                dbName, sql_statement, one_episode = backlog.pop(0)
                new_fut = actor.test_build_sql.remote(dbName, sql_statement)
                pending[new_fut] = (actor, dbName, one_episode)
        # self.sql_buffer.update_priorities()
        self.pending_state = pending

    def _try_collect_sql(self, fut, dbName, one_episode):
        try:
            sql_obj: SQL = ray.get(fut)
            if sql_obj is not None:
                self.logger.info(f"[SQLGenTrainerActor] dbName: {sql_obj.dbName}\n sql_statement: {sql_obj.sql_statement}\n min_latency: {sql_obj.min_latency:.2f}, max_latency: {sql_obj.max_latency:.2f}, base_latency: {sql_obj.base_latency:.2f}")
                sql_obj.update_reward(0.0)
                for step in one_episode:
                    self.sqlgen_agent.store_transition(*step)
                pos = self.sqlgen_agent.get_buffer_position()
                self.sql_buffer.push(sql_obj, pos)
        except Exception as e:
            self.logger.error(f"[SQLGenTrainerActor] Error processing task: {e}")
            
    def _generate_one_sql(self, dbName: str):
        self.logger.info(f"[SQLGenTrainerActor] generate_one_sql for dbName={dbName}")
        state, _ = self.sqlgen_env.reset(options={'dbName': dbName})
        action_mask = state['action_mask']
        done = False
        one_episode = []
        while not done:
            action = self.sqlgen_agent.select_action(state, action_mask)
            next_state, reward, done, _, _ = self.sqlgen_env.step(action)
            next_action_mask = next_state['action_mask']
            one_episode.append((state, action, reward, next_state, done, action_mask, next_action_mask))
            state = next_state
            action_mask = next_action_mask
        sql_statement = self.sqlgen_env.get_query()
        
        return sql_statement, one_episode

    def _update_sqlgen_reward(self, buffer_contents):
        for idx, sql in enumerate(buffer_contents): 
            sql:SQL
            if sql.dbName not in self.test_databases:
                self.sqlgen_agent.set_step_reward(sql.sqlgen_reward, sql.pos_in_sqlgen_buffer)
            
    def _update_sqlgen_model(self):
        iteration = math.ceil(math.log(len(self.sqlgen_agent.replay_buffer)) * 20) 
        self.logger.info(f"[SQLGenTrainerActor] - Buffer Size: {len(self.sqlgen_agent.replay_buffer)}, Iteration: {iteration}")
        q1_loss, q2_loss, policy_loss, alpha_loss = self.sqlgen_agent.update_parameters(iteration) # Old way
        self.logger.info(f"[SQLGenTrainerActor] - Q1: {q1_loss:.4f}, Q2: {q2_loss:.4f}, Policy: {policy_loss:.4f}, Alpha: {alpha_loss:.4f}")

class LQOTrainerActor:
    def __init__(self, sql_buffer:SQLBuffer, seed=1408, load_pretrain = False):
        set_seed(seed)
        self.lqo_config = LQOConfig()
        self.lqo_test_env = LQOEnvTest(self.lqo_config)
        self.lqo_collect_env = LQOEnvExp()

        self.sql_buffer:SQLBuffer = sql_buffer
        self.test_sql = self._load_test_sql(self.lqo_config.test_sql_path)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def iterate_one_loop(self):
        all_sql, _ = self.sql_buffer.get_sql_buffer()
        all_training_data = self.collect_data(all_sql)
        train_dataset = QDataset(all_training_data)
        self.lqo_agent = LQOAgent(self.lqo_config)
        final_losses = self.lqo_agent.update(train_dataset)
        self._evaluate_all_sql(self.sql_buffer, self.lqo_test_env, final_losses)
        self._test_sql(self.test_sql, self.lqo_test_env)
        self.lqo_agent.save_model()

    def collect_data(self, all_sql):
        all_experiences = []
        for sql in all_sql:
            sql:SQL
            experiences = self._collect_experiences_for_query(self.lqo_collect_env, sql)
            all_experiences.extend(experiences)
        print(f"Total experiences collected: {len(all_experiences)}")
        return all_experiences
    
    def _collect_experiences_for_query(self, env: LQOEnvExp, sql_instance: SQL):
        experiences = []
        initial_state, plan_state = env.reset(options = {'sql': sql_instance})
        root = Node(initial_state, plan_state)
        visited_nodes = set()
        def expand_node(node: Node):
            if node.code in visited_nodes:
                return
            visited_nodes.add(node.code)
            original_env_state = env.save_env_state()
            valid_actions = np.where(node.env_state['action_mask'] == 1)[0].tolist()
            rewards = np.zeros(len(HINT2POS), dtype=np.float16)
            for action in range(len(HINT2POS)):
                if action not in valid_actions:
                    rewards[action] = -1.0
                    continue
                env.load_env_state(original_env_state)
                next_state, reward, done, _, plan_state = env.step(action)
                rewards[action] = reward
                child = Node(next_state, plan_state, action, node)
                child.code = env.current_code
                node.children[action] = child
                if not done:
                    expand_node(child)
            # rewards = rewards
            # node.env_state['action_mask'] = node.env_state['action_mask']
            # node.env_state['action_code'] = node.env_state['action_code']
            if len(node.env_state['heights']) <= MAXNODE:
                experiences.append((node.env_state, rewards, sql_instance.pos_in_sqlgen_buffer))
            env.load_env_state(original_env_state)
        expand_node(root)
        return experiences
    
    def normalize_rewards(self, all_experiences):
        all_rewards = []
        for state, rewards, _ in all_experiences:
            all_rewards.extend(rewards)
        all_rewards = np.array(all_rewards)
        positive_mask = all_rewards > 0
        negative_mask = all_rewards < 0
        
        positive_rewards = all_rewards[positive_mask]
        negative_rewards = all_rewards[negative_mask]
        normalized_experiences = []
        min_pos = np.min(positive_rewards)
        max_pos = np.max(positive_rewards)
        min_neg = np.min(negative_rewards)
        max_neg = np.max(negative_rewards)

        for state, rewards,pos_in_sqlgen_buffer in all_experiences:
            normalized_rewards = np.zeros_like(rewards)
            for i, reward in enumerate(rewards):
                if reward > 0 and len(positive_rewards) > 0:
                    
                    if max_pos > min_pos:
                        normalized_rewards[i] = (reward - min_pos) / (max_pos - min_pos)
                    else:
                        normalized_rewards[i] = 1.0
                elif reward < 0 and len(negative_rewards) > 0:
                    if min_neg < max_neg:
                        normalized_rewards[i] = (reward - max_neg) / (max_neg - min_neg)
                    else:
                        normalized_rewards[i] = -1.0
                else:
                    normalized_rewards[i] = 0.0
            
            normalized_experiences.append((state, normalized_rewards, pos_in_sqlgen_buffer))
        return normalized_experiences
    
    def _load_test_sql(self, test_sql_path):
        test_sql = []
        for test_sql_path in test_sql_path:
            if os.path.exists(test_sql_path):   
                test_job_sql = pickle.load(open(test_sql_path, 'rb'))
                if test_sql_path.endswith('test_jobext_sql.pkl'):
                    for sql in test_job_sql:
                        sql.dbName = 'imdb_ext'
                test_sql.extend(test_job_sql)
        return test_sql
    
    def _evaluate_one_sql(self, sql: SQL, eval_env: LQOEnvTest, cal_sqlgen_reward = True):
        obs, _ = eval_env.reset(options={'sql': sql})
        done = False
        while not done:
            obs['action_code'] = obs['action_code']
            q_values = self.lqo_agent.predict(obs)
            action_mask = obs['action_mask']
            masked_q_values = np.where(action_mask == 1, q_values, -np.inf)
            if np.all(masked_q_values <= 0.0):
                done = True
            else:
                action = np.argmax(masked_q_values)
                obs, latency, done, _, _ = eval_env.step(action)
        if cal_sqlgen_reward:
            reward = self._cal_sqlgen_reward(sql, latency)
        else:
            reward = latency
        return reward, latency
    
    def _cal_sqlgen_reward(self, sql: SQL, latency: float):
        base_latency = sql.base_latency
        min_latency = sql.q_min_latency
        max_latency = sql.max_latency
        if abs(base_latency - sql.min_latency) / base_latency < 0.01: 
            if abs(base_latency - latency) / base_latency < 0.01:  
                sqlgen_reward = 0.0
            else:
                sqlgen_reward = (latency - base_latency) / (max_latency - base_latency)
        else:
            if latency > base_latency:  
                sqlgen_reward = (latency - base_latency) / (2 * (max_latency - base_latency)) + 0.5
            else:
                sqlgen_reward = np.clip((latency - min_latency) / (2 * (base_latency - min_latency + 1e-8)), 0.0, 0.5)  # latency
        weight = max(5 * math.log2(base_latency + 1) / math.log2(PLANMAXTIMEOUT), 1) 
        sqlgen_reward *= weight
        return sqlgen_reward
    
    def _get_q_min_latency(self, sql: SQL, eval_env: LQOEnvTest):
        obs, _ = eval_env.reset(options={'sql': sql})
        done = False
        while not done:
            action_mask = obs['action_mask']
            max_idx = -1
            now_latency = sql.get_plan_by_hint_code(eval_env.current_code).latency
            for idx, value in enumerate(action_mask):
                if value == 1:
                    plan: Plan = sql.get_plan_by_hint_code(eval_env.current_code + pow(2, idx))
                    if plan.latency < now_latency:
                        max_idx = idx
                        now_latency = plan.latency
            if max_idx == -1:
                done = True
            else:
                obs, reward, done, _, latency = eval_env.step(max_idx)
        return now_latency
    
    def _evaluate_all_sql(self, sql_buffer: SQLBuffer, eval_env: LQOEnvTest, final_losses: dict = None):
        # Get all SQLs from the buffer
        buffer_contents, _ = sql_buffer.get_sql_buffer()
        buffer_size = len(buffer_contents)
        all_rewards = []
        all_min_latencies = 0.0
        all_base_latencies = 0.0
        all_lqo_latencies = 0.0
        all_q_min_latencies = 0.0
        for idx in range(buffer_size):
            sql:SQL = buffer_contents[idx]
            if sql is not None:
                if not hasattr(sql, 'q_min_latency') or sql.q_min_latency is None:
                    sql.q_min_latency = self._get_q_min_latency(sql, eval_env)
                sqlgen_reward, latency = self._evaluate_one_sql(sql, eval_env)
                if final_losses is not None:
                    sqlgen_reward += final_losses.get(sql.pos_in_sqlgen_buffer, 0.0)
                sql_buffer.update_sql_reward(idx, sqlgen_reward)
                all_rewards.append(sqlgen_reward)
                all_min_latencies += sql.min_latency
                all_base_latencies += sql.base_latency
                all_lqo_latencies += latency
                all_q_min_latencies += sql.q_min_latency
        avg_reward =  sum(all_rewards) / buffer_size
        improve_ratio = (all_lqo_latencies - all_q_min_latencies) / (all_base_latencies - all_q_min_latencies)
        self.logger.info(f"[LQOTrainerActor] evaluate_all_sql: buffer_size={buffer_size}, SQLGen avg_reward={avg_reward:.4f} LQO improve_ratio={improve_ratio:.4f}")
        self.logger.info(f"[LQOTrainerActor] evaluate_all_sql: min_latency={all_min_latencies / 1000:.4f}, q_min_latency={all_q_min_latencies / 1000:.4f}, lqo_latency={all_lqo_latencies / 1000:.4f}, base_latency={all_base_latencies / 1000:.4f}")
        return improve_ratio
    
    def _test_sql(self, test_sql, test_env):
        min_latency = {}
        base_latency = {}
        lqo_reward = {}
        q_min_latency = {}
        for sql in test_sql:
            sql:SQL
            if not hasattr(sql, 'q_min_latency') or sql.q_min_latency is None:
                sql.q_min_latency = self._get_q_min_latency(sql, test_env)
            reward, latency = self._evaluate_one_sql(sql, test_env, cal_sqlgen_reward = False)
            if sql.dbName not in min_latency:
                min_latency[sql.dbName] = 0
                base_latency[sql.dbName] = 0
                lqo_reward[sql.dbName] = 0
                q_min_latency[sql.dbName] = 0
            min_latency[sql.dbName] += sql.min_latency
            base_latency[sql.dbName] += sql.base_latency
            q_min_latency[sql.dbName] += sql.q_min_latency
            lqo_reward[sql.dbName] += reward
        total_reward = 0
        total_min_latency = 0
        total_base_latency = 0
        total_q_min_latency = 0
        for dbName in min_latency:
            total_reward += lqo_reward[dbName]
            if dbName in self.lqo_config.test_databases:
                self.logger.info(f"[LQOTrainerActor] Test: dbName={dbName}, min_latency={min_latency[dbName] / 1000:.3f}, q_min_latency={q_min_latency[dbName] / 1000:.3f}, base_latency={base_latency[dbName] / 1000:.3f}, lqo_latency={lqo_reward[dbName] / 1000:.3f}")
            else:
                self.logger.info(f"[LQOTrainerActor] Train: dbName={dbName}, min_latency={min_latency[dbName] / 1000:.3f}, q_min_latency={q_min_latency[dbName] / 1000:.3f}, base_latency={base_latency[dbName] / 1000:.3f}, lqo_latency={lqo_reward[dbName] / 1000:.3f}")
            total_min_latency += min_latency[dbName]
            total_q_min_latency += q_min_latency[dbName]
            total_base_latency += base_latency[dbName]
        self.logger.info(f"[LQOTrainerActor] total_min_latency={total_min_latency / 1000:.3f}, total_q_min_latency={total_q_min_latency / 1000:.3f}, total_base_latency={total_base_latency / 1000:.3f}, total_reward={total_reward / 1000:.3f}")
        return min_latency, base_latency, lqo_reward
    
def main():
    ray.init(log_to_driver=True)
    seed = 9999
    sql_buffer = SQLBuffer()
    if os.path.exists('./ckpt/sql_buffer.pkl'):
        sql_buffer.load_state('./ckpt/sql_buffer.pkl')
    sqlgen_trainer = SQLGenTrainerActor(sql_buffer = sql_buffer, seed = seed, load_pretrain = False)
    lqo_trainer = LQOTrainerActor(sql_buffer = sql_buffer, seed = seed, load_pretrain = False)
    sqlgen_trainer.generate_N_sql(num_generated_sql = 10)
    while True:
        lqo_trainer.iterate_one_loop() 
        sqlgen_trainer.iterate_one_loop(60)
        sql_buffer.save_state(sqlgen_trainer.sqlgen_config.checkpoint_dir)

if __name__ == "__main__":

    main()