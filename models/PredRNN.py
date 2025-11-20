import torch
import torch.nn as nn
from layers.SpatioTemporalLSTMCell import SpatioTemporalLSTMCell


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.configs = configs
        self.frame_channel = configs.input_channels
        self.num_layers = configs.num_layers
        self.num_hidden = configs.num_hidden
        self.kernel_size = configs.kernel_size
        cell_list = []

        width = configs.grid_size[0] 

        # Channel groups info
        self.num_std = getattr(configs, 'num_std', 0)
        self.num_minmax = getattr(configs, 'num_minmax', 0)
        self.num_robust = getattr(configs, 'num_robust', 0)
        self.num_tcc = getattr(configs, 'num_tcc', 0)
        
        self.std_indices = getattr(configs, 'std_cols_indices', [])
        self.minmax_indices = getattr(configs, 'minmax_cols_indices', [])
        self.robust_indices = getattr(configs, 'robust_cols_indices', [])
        self.tcc_indices = getattr(configs, 'tcc_cols_indices', []) 

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width, self.kernel_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        
        # 4 heads
        self.head_std = nn.Conv2d(self.num_hidden[-1], self.num_std, kernel_size=1, stride=1, padding=0) if self.num_std > 0 else None
        self.head_minmax = nn.Conv2d(self.num_hidden[-1], self.num_minmax, kernel_size=1, stride=1, padding=0) if self.num_minmax > 0 else None
        self.head_robust = nn.Conv2d(self.num_hidden[-1], self.num_robust, kernel_size=1, stride=1, padding=0) if self.num_robust > 0 else None
        self.head_tcc = nn.Conv2d(self.num_hidden[-1], self.num_tcc, kernel_size=1, stride=1, padding=0) if self.num_tcc > 0 else None  

    def _apply_heads(self, hidden_state):
        outputs = []
        indices = []
        
        if self.head_std:
            outputs.append(self.head_std(hidden_state))
            indices.extend(self.std_indices)
        if self.head_minmax:
            outputs.append(self.head_minmax(hidden_state))
            indices.extend(self.minmax_indices)
        if self.head_robust:
            outputs.append(self.head_robust(hidden_state))
            indices.extend(self.robust_indices)
        if self.head_tcc:
            outputs.append(self.head_tcc(hidden_state))
            indices.extend(self.tcc_indices)
            
        # Reassemble
        full_output = torch.cat(outputs, dim=1)     
        sort_idx = torch.argsort(torch.tensor(indices)).to(full_output.device)
        return full_output[:, sort_idx, :, :]   


    def forward(self, frames_past, mask_true=None, true_frames=None):
        frames = frames_past.contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        input_length = frames.shape[1]
        
        total_sim_steps = self.configs.his_len + self.configs.pred_len
        his_len = self.configs.his_len

        next_frames = []
        h_t = []; c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.gpu)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.gpu)

        # Vòng lặp chính
        for t in range(total_sim_steps):
            if  self.configs.reverse_scheduled_sampling:
                if t == 0:
                    net = frames_past[:, t]
                else:
                    if mask_true is not None and true_frames is not None:
                        mask = mask_true[:, t]
                        if t < his_len:
                            true_frame = frames_past[:, t]
                        else:
                            true_frame = true_frames[:, t - his_len]
                        net = mask * true_frame + (1 - mask) * x_gen
                    else:
                        if t < his_len:
                            net = frames_past[:, t]
                        else:
                            net = x_gen
            else:
                if t < input_length:
                    net = frames_past[:, t]
                else:
                    net = x_gen 
            
            # Cập nhật SpatioTemporal LSTM Cell
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self._apply_heads(h_t[self.num_layers - 1])
            
            if t >= input_length:
                next_frames.append(x_gen)

        # [batch, L_pred, H, W, C]
        next_frames = torch.stack(next_frames, dim=0).contiguous()
        next_frames = next_frames.permute(1, 0, 2, 3, 4)  # [batch, length, C, H, W]
        # print(f"Shape next_frames: {next_frames.shape}")
        
        return next_frames