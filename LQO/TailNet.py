import torch
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import torch.nn as nn
from ray.rllib.utils.torch_utils import FLOAT_MIN
from LQO.config import Config
from LQO.constant import *
from LQO.PlanNet import PlanNetwork
config = Config()

class PairModel(nn.Module):
    def __init__(self):
        super(PairModel, self).__init__()
        self.planNet = PlanNetwork()
        
        # Comparison network that takes two plan embeddings and outputs which plan is better
        self.comparison_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
    
    def get_plan_embedding(self, x):
        """Extract plan embedding using the planNet"""
        return self.planNet(x)[:, 0, :]
    
    def forward(self, plan1, plan2):
        """
        Compare two query plans and return which one is better
        
        Args:
            plan1: Features of the first plan
            plan2: Features of the second plan
            
        Returns:
            Probability distribution: [p(plan1 is better), p(plan2 is better)]
        """
        # Process both plans through the network to get embeddings
        plan1_embedding = self.get_plan_embedding(plan1)
        plan2_embedding = self.get_plan_embedding(plan2)
        
        # Concatenate the embeddings
        combined_embedding = torch.cat([plan1_embedding, plan2_embedding], dim=1)
        
        # Get comparison result (which plan is better)
        comparison_result = self.comparison_mlp(combined_embedding)
        
        return comparison_result

class PointModel(nn.Module):
    def __init__(self):
        super(PointModel, self).__init__()
        self.planNet = PlanNetwork()
        self.card_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.cost_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        embedding = self.planNet(x)
        cost = self.cost_head(embedding)
        card = self.card_head(embedding)
        return cost, card, embedding

class QMultiNetwork(nn.Module):
    def __init__(self):
        super(QMultiNetwork, self).__init__()

        self.planStateEncoder = PlanNetwork()
        # Action code processing
        self.actionStateEncoder = nn.Sequential(
            nn.Linear(6, 16),  # 7 is action_code dimension
            nn.GELU()
        )

        self.qhead = nn.Sequential(
            nn.LayerNorm(self.planStateEncoder.hidden_dim + 16),
            nn.Linear(self.planStateEncoder.hidden_dim + 16, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.mlp_dropout),
            nn.Linear(config.mlp_dim, config.mlp_dim // 2),
            nn.GELU(),
            nn.Linear(config.mlp_dim // 2, config.mlp_dim // 4),
            nn.GELU(),
            nn.Linear(config.mlp_dim // 4, 6),
        )
        # self.lhead = nn.Sequential(
        #     nn.LayerNorm(self.planStateEncoder.hidden_dim),
        #     nn.Linear(self.planStateEncoder.hidden_dim, 128),
        #     nn.LeakyReLU(),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(128, 64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 32),
        #     nn.LeakyReLU(),
        #     nn.Linear(32, 1)
        # )
    def getStateVector(self, state):
        embed_plan = {
            "x": state["x"],
            "attn_bias": state["attn_bias"],
            "heights": state["heights"]
        }
        planStateFeatures = self.planStateEncoder(embed_plan)[:, 0, :]
        return planStateFeatures

    #     planStateFeatures = self.planStateEncoder(embed_plan)[:, 0, :]
    #     actionStateFeatures = self.actionStateEncoder(state["action_code"])
    #     stateFeatures = torch.cat([planStateFeatures, actionStateFeatures], dim=1)
    #     stateVector = self.stateEncoder(stateFeatures)
    #     return stateVector

    def forward(self, state):
        embed_plan = {
            "x": state["x"],
            "attn_bias": state["attn_bias"],
            "heights": state["heights"]
        }

        planStateFeatures = self.planStateEncoder(embed_plan)[:, 0, :]
        actionStateFeatures = self.actionStateEncoder(state["action_code"])
        stateFeatures = torch.cat([planStateFeatures, actionStateFeatures], dim=1)
        values = self.qhead(stateFeatures)
        # latency = self.lhead(planStateFeatures)
        return values
    
class QNetwork(nn.Module):
    def __init__(self,feature_dict):
        super(QNetwork, self).__init__()

        self.action_embed_dim = 16


        self.planStateEncoder = PlanNetwork(feature_dict)
        # Action code processing
        self.actionStateEncoder = nn.Sequential(
            nn.Linear(7, 16),  # 7 is action_code dimension
            nn.LayerNorm(16),
            nn.LeakyReLU()
        )

        self.stateEncoder = nn.Sequential(
            nn.Linear(self.planStateEncoder.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU()
        )

        self.action_embed = nn.Embedding(7, self.action_embed_dim)

        self.qhead = nn.Sequential(
            nn.Linear(128 + self.action_embed_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            # nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )
        
    def getStateVector(self, state):
         # Process plan features
        embed_plan = {
            "x": state["x"],
            "attn_bias": state["attn_bias"],
            "heights": state["heights"]
        }

        planStateFeatures = self.planStateEncoder(embed_plan)

        # actionStateFeatures = self.actionStateEncoder(state["action_code"])
        # stateFeatures = torch.cat([planStateFeatures, actionStateFeatures], dim=1)
        stateVector = self.stateEncoder(planStateFeatures)
        return stateVector
    
    def getActionVector(self, action_code):
        actionVector = self.action_embed(action_code)
        return actionVector

    def forward(self, state, action_code):

        stateVector = self.getStateVector(state)
        actionVector = self.getActionVector(action_code)
        combined = torch.cat([stateVector, actionVector], dim=1)
        values = self.qhead(combined)
        return values