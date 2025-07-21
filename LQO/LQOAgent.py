import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LQO.config import Config
from LQO.TailNet import QMultiNetwork
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
# from tqdm import tqdm
class QDataset(Dataset):
    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        state, rewards, pos = self.experiences[idx]
        
        state_tensors = {}
        for k, v in state.items():
            dtype = torch.float32
            if 'mask' in k:
                dtype = torch.bool if k != 'action_mask' else torch.float32

            if isinstance(v, np.ndarray):
                state_tensors[k] = torch.from_numpy(v).to(dtype)
            else:
                state_tensors[k] = torch.tensor(v, dtype=dtype)

        return state_tensors, torch.tensor(rewards, dtype=torch.float32), torch.tensor(pos, dtype=torch.int32)
    
class LQOAgent:
    def __init__(self, config:Config):
        self.config = config
        self.q_network = QMultiNetwork().to(self.config.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)
        self.model_save_path = config.lqo_agent_path

    def update(self, train_dataset, num_epochs=100, batch_size=1024):

        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        self.q_network.train()
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            
            # progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
            for batch_data in data_loader:
                states, target_q_values, pos = batch_data
                
                states = {k: v.to(self.config.device) for k, v in states.items()}
                target_q_values = target_q_values.to(self.config.device)
                predicted_q_values = self.q_network(states)
                
                loss_mask = states['action_mask'].bool()
                squared_error = (predicted_q_values - target_q_values) ** 2
                masked_error = squared_error * loss_mask.float()
                
                if loss_mask.sum() > 0:
                    loss = masked_error.sum() / loss_mask.sum()
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.config.device)
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                
                total_loss += loss.item()
                # progress_bar.set_postfix({
                #     'total_loss': f'{loss.item():.6f}',
                # })
        
            avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
            if len(losses) > 15 and losses[-1] < 0.01:
                last_two = np.min(losses[-2:])
                if last_two > losses[-5] or (losses[-5] - last_two < 0.001):
                    print("Stopped training from convergence condition at epoch", epoch)
                    break
        final_losses = self.obtain_loss_per_pos(data_loader)
        return final_losses
    
    def obtain_loss_per_pos(self, data_loader):
        self.q_network.eval()
        losses = {}
        for batch_data in data_loader:
            states, target_q_values, pos = batch_data
            states = {k: v.to(self.config.device) for k, v in states.items()}
            target_q_values = target_q_values.to(self.config.device)
            with torch.no_grad():
                predicted_q_values = self.q_network(states)
            loss_mask = states['action_mask'].bool()
            squared_error = (predicted_q_values - target_q_values) ** 2
            masked_error = squared_error * loss_mask.float()
            
            for i in range(len(pos)):
                sample_mask = loss_mask[i]
                if sample_mask.sum() > 0:
                    sample_loss = masked_error[i][sample_mask].mean()
                else:
                    sample_loss = torch.tensor(0.0).to(self.config.device)
                if pos[i].item() not in losses: 
                    losses[pos[i].item()] = []
                losses[pos[i].item()].append(sample_loss.cpu().item())
        for pos in losses:
            losses[pos] = np.mean(losses[pos])
        
        if losses:
            loss_values = list(losses.values())
            min_loss = min(loss_values)
            max_loss = max(loss_values)
            print(f'min_loss: {min_loss}, max_loss: {max_loss}',flush=True)

            if max_loss - min_loss > 0:
                for pos in losses:
                    normalized_loss = 5 * (losses[pos] - min_loss) / (max_loss - min_loss)
                    losses[pos] = normalized_loss
            else:
                for pos in losses:
                    losses[pos] = 2.5
        
        return losses
            
    def save_model(self):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.model_save_path, 'lqo_agent.pth'))
        print(f"Model saved to {self.model_save_path}")

    def load_model(self):

        if not os.path.exists(self.model_save_path):
            print(f"No model found at {self.model_save_path}. Starting from scratch.")
            return
        model_path = os.path.join(self.model_save_path, 'lqo_agent.pth')
        try:
            checkpoint = torch.load(model_path, map_location=self.config.device)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            if k == 'step':
                                state[k] = v.cpu()
                            else:
                                state[k] = v.to(self.config.device)
            self.q_network.to(self.config.device)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting from scratch.")

    def predict(self,state):
        self.q_network.eval()
        model_input = {
            k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(self.config.device)
            for k, v in state.items() if k in ["x", "attn_bias", "heights", "action_code"]
        }
        with torch.no_grad():
            q_values = self.q_network(model_input)
        return q_values.cpu().numpy()
    