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

        for i in range(self.num_layers):
            in_channel = self.frame_channel if i == 0 else self.num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, self.num_hidden[i], width, self.kernel_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(self.num_hidden[self.num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, frames_past):
        frames = frames_past.contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]
        input_length = frames.shape[1]
        
        total_sim_steps = self.configs.his_len + self.configs.pred_len - 1 

        next_frames = []
        h_t = []; c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.gpu)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.gpu)

        # Vòng lặp chính
        for t in range(total_sim_steps):
            if t < input_length:
                net = frames[:, t]
            else:
                net = x_gen 
            
            # Cập nhật SpatioTemporal LSTM Cell
            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            
            if t >= input_length - 1:
                next_frames.append(x_gen)

        # [batch, L_pred, H, W, C]
        next_frames = torch.stack(next_frames, dim=0).contiguous()
        next_frames = next_frames.permute(1, 0, 2, 3, 4)  # [batch, length, C, H, W]
        # print(f"Shape next_frames: {next_frames.shape}")
        
        return next_frames