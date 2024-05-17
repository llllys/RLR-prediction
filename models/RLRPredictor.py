import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder
from .agentformer_lib import AgentFormerEncoderLayer, AgentFormerEncoder
from .HybridCrossAttention import HybridCrossAttentionEncoder, HybridCrossAttentionEncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        d_model = args.model_dim
        dropout = args.dropout
        max_len = 5000
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.linear = nn.Linear(2*d_model, d_model)
        
    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = torch.concat([x, self.pe[:, :x.size(1), :].repeat(x.size(0), 1, 1)], axis=2)
        x = self.linear(x)
        # return self.dropout(x)
        return x


class PositionalAgentEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_t_len=200, max_a_len=200, concat=False, use_agent_enc=False, agent_enc_learn=False):
        super(PositionalAgentEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        self.use_agent_enc = use_agent_enc
        if concat:
            self.fc = nn.Linear((3 if use_agent_enc else 2) * d_model, d_model)

        pe = self.build_pos_enc(max_t_len)
        self.register_buffer('pe', pe)
        if use_agent_enc:
            if agent_enc_learn:
                self.ae = nn.Parameter(torch.randn(max_a_len, 1, d_model) * 0.1)
            else:
                ae = self.build_pos_enc(max_a_len)
                self.register_buffer('ae', ae)

    def build_pos_enc(self, max_len):
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        return pe

    def build_agent_enc(self, max_len):
        ae = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-np.log(10000.0) / self.d_model))
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0).transpose(0, 1)
        return ae
    
    def get_pos_enc(self, num_t, num_a, t_offset):
        pe = self.pe[t_offset: num_t + t_offset, :]
        pe = pe.repeat_interleave(num_a, dim=0)
        return pe

    def get_agent_enc(self, num_t, num_a, a_offset, agent_enc_shuffle):
        if agent_enc_shuffle is None:
            ae = self.ae[a_offset: num_a + a_offset, :]
        else:
            ae = self.ae[agent_enc_shuffle]
        ae = ae.repeat(num_t, 1, 1)
        return ae

    def forward(self, x, num_a, agent_enc_shuffle=None, t_offset=0, a_offset=0):
        num_t = x.shape[0] // num_a
        pos_enc = self.get_pos_enc(num_t, num_a, t_offset)
        if self.use_agent_enc:
            agent_enc = self.get_agent_enc(num_t, num_a, a_offset, agent_enc_shuffle)
        if self.concat:
            feat = [x, pos_enc.repeat(1, x.size(1), 1)]
            if self.use_agent_enc:
                feat.append(agent_enc.repeat(1, x.size(1), 1))
            x = torch.cat(feat, dim=-1)
            x = self.fc(x)
        else:
            x += pos_enc
            if self.use_agent_enc:
                x += agent_enc
        return self.dropout(x)


class RLRPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        
        # input 
        self.in_env_dim = args.input_env_dim
        self.in_ego_dim = args.input_ego_dim
        self.in_nbr_dim = args.input_nbr_dim
        self.input_length = args.input_length

        # output
        self.n_modes = args.K
        self.pred_GMM = args.pred_GMM
        self.pred_traj_dim = args.pred_dim
        self.pred_intention_dim = args.pred_intention_dim
        self.output_length = args.output_length

        # model 
        self.model_dim = args.model_dim
        self.ff_dim = args.dim_ff
        self.nhead = args.num_heads
        self.dropout = args.dropout
        self.nlayer = args.num_layers
        
        # agent-wise attention
        self.agent_enc_shuffle = args.agent_enc_shuffle
        self.pos_concat = args.pos_concat
        self.max_agent_len = args.max_agent_len
        self.use_agent_enc = args.use_agent_enc
        self.agent_enc_learn = args.agent_enc_learn

        self.input_nbr_fc = nn.Linear(self.in_nbr_dim, self.model_dim)
        self.input_ego_fc = nn.Linear(self.in_ego_dim, self.model_dim)
        self.input_env_fc = nn.Linear(self.in_env_dim, self.model_dim)
        self.context_fc = nn.Linear(self.model_dim, self.model_dim)

        # encoder
        self.agent_pos_encoder = PositionalAgentEncoding(self.model_dim,
                                                         self.dropout, 
                                                         concat=self.pos_concat, 
                                                         max_a_len=self.max_agent_len, 
                                                         use_agent_enc=self.use_agent_enc, 
                                                         agent_enc_learn=self.agent_enc_learn)
        self.pos_enc_ego = PositionalEncoding(args)
        self.pos_enc_env = PositionalEncoding(args)

        # 1st level
        encoder_layers = AgentFormerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout)
        self.af_encoder = AgentFormerEncoder(encoder_layers, self.nlayer)
        tf_encoder_layers_env = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout, batch_first=True)
        self.tf_encoder_env = TransformerEncoder(tf_encoder_layers_env, self.nlayer)
        tf_encoder_layers_ego = TransformerEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout, batch_first=True)
        self.tf_encoder_ego = TransformerEncoder(tf_encoder_layers_ego, self.nlayer)

        # 2nd level
        HCA_layer_env_nbr = HybridCrossAttentionEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout, batch_first=True)
        self.HCA_env_nbr = HybridCrossAttentionEncoder(HCA_layer_env_nbr, self.nlayer)
        HCA_layer_ego_nbr = HybridCrossAttentionEncoderLayer(self.model_dim, self.nhead, self.ff_dim, self.dropout, batch_first=True)
        self.HCA_ego_nbr = HybridCrossAttentionEncoder(HCA_layer_ego_nbr, self.nlayer)
        
        # 3rd level
        self.decoder_env_MHA = nn.MultiheadAttention(self.model_dim, self.nhead, self.dropout, batch_first=True)
        self.decoder_ego_MHA = nn.MultiheadAttention(self.model_dim, self.nhead, self.dropout, batch_first=True)
        
        # decoder
        self.anchor_decode_fc = nn.Linear(self.model_dim, self.n_modes * self.model_dim * self.output_length)
        self.traj_dec_fc = nn.Linear(2 * self.model_dim * self.output_length, self.pred_traj_dim * self.output_length + 1)
        self.intention_dec_fc = nn.Linear((2 * self.model_dim + self.pred_traj_dim) * self.output_length, self.pred_intention_dim * self.output_length)
    
    def cal_mask_input(self, data):

        agent_num = data['neighbor_motion'].shape[1]
        
        nbr_pos = data['neighbor_motion'][:, :, :, 0:2].unsqueeze(1).repeat(1, self.nhead, 1, 1, 1).reshape(-1, agent_num, self.input_length, 2) # [B*head, N, L, 2]
        nbr_pos = torch.permute(nbr_pos, (0, 2, 1, 3)).reshape(-1, agent_num * self.input_length, 2) # [Bh, NL, 2]
        nbr_pos1 = nbr_pos.unsqueeze(1).repeat((1, agent_num * self.input_length, 1, 1))
        nbr_pos2 = nbr_pos.unsqueeze(2).repeat((1, 1, agent_num * self.input_length, 1))
        dist_diff = nbr_pos1 - nbr_pos2
        dist_diff = torch.norm(dist_diff, p=2, dim=-1)
        max_dist_diff, _ = dist_diff.max(dim=1)
        max_dist_diff, _ = max_dist_diff.max(dim=1)
        dist_diff = dist_diff / (max_dist_diff[:, None, None] + 1e-4)
        dist_diff = dist_diff.unsqueeze(0)
        
        nbr_heading = data['neighbor_heading'].unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(-1, agent_num, self.input_length) # [B*head, N, L]
        nbr_heading = torch.permute(nbr_heading, (0, 2, 1)).reshape(-1, agent_num * self.input_length)
        nbr_heading1 = nbr_heading.unsqueeze(1).repeat((1, agent_num * self.input_length, 1))
        nbr_heading2 = nbr_heading.unsqueeze(2).repeat((1, 1, agent_num * self.input_length))
        heading_diff = (torch.cos((nbr_heading1 - nbr_heading2).abs() / 360.0 * math.pi * 2) + 1.0) / 2.0 # [512, 72, 72]
        heading_diff = heading_diff.unsqueeze(0)

        nbr_speed = data['neighbor_motion'][:, :, :, 2].unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(-1, agent_num, self.input_length) # [B*head, N, L]
        nbr_speed = torch.permute(nbr_speed, (0, 2, 1)).reshape(-1, agent_num * self.input_length)
        nbr_speed1 = nbr_speed.unsqueeze(1).repeat((1, agent_num * self.input_length, 1))
        nbr_speed2 = nbr_speed.unsqueeze(2).repeat((1, 1, agent_num * self.input_length))
        speed_diff = (nbr_speed1 - nbr_speed2).abs() # [512, 72, 72]
        max_speed_diff, _ = speed_diff.max(dim=1)
        max_speed_diff, _ = max_speed_diff.max(dim=1)
        speed_diff = speed_diff / (max_speed_diff[:, None, None] + 1e-4)
        speed_diff = speed_diff.unsqueeze(0)

        pos_mask_in = torch.concat([dist_diff, speed_diff, heading_diff], axis=0) # [3, 512, 72, 72]
        
    def intention2traj_decoder(self, out, data):
        '''
            out: [B, L, D]
        '''
        predictions = out[:, -1, :]
        
        dec_anchor = self.anchor_decode_fc(predictions).reshape(out.shape[0], self.n_modes, self.output_length, self.model_dim)
        out_repeat_mode = out[:, -1, :].reshape(out.shape[0], 1, 1, self.model_dim).repeat(1, self.n_modes, self.output_length, 1)
        dec_anchor_traj = torch.concat([out_repeat_mode, dec_anchor], axis=3).reshape(out.shape[0], self.n_modes, -1) # [B, K, LD]
        
        dec_traj = self.traj_dec_fc(dec_anchor_traj).reshape(out.shape[0], self.n_modes, -1) # [B, K, L*D+1]
        dec_traj_probas = dec_traj[:, :, -1]
        dec_traj = dec_traj[:, :, :-1].reshape(out.shape[0], self.n_modes, self.output_length, self.pred_traj_dim)

        dec_anchor_intention = torch.concat([out_repeat_mode, dec_anchor, dec_traj], axis=3).reshape(out.shape[0], self.n_modes, -1)
        dec_intention = self.intention_dec_fc(dec_anchor_intention).reshape(out.shape[0], self.n_modes, self.output_length, self.pred_intention_dim)
        dec_intention_whole = dec_intention.mean(axis=2) 

        ret_dict = {}
        ret_dict['pre_is_rlr_whole'] = dec_intention_whole
        ret_dict['pre_is_rlr_step'] = dec_intention
        ret_dict['pre_motion'] = dec_traj
        ret_dict['pre_motion_probas'] = dec_traj_probas
        
        return ret_dict


    def forward(self, data, mode='train'):
        '''
            ego_motion: [B, L, D]
            input_env: [B, L, D]
            nbr_motion: [B, N, L, D]
        '''
        ego_motion = data['ego_motion_x']
        ego_feature = data['ego_motion_feature']
        input_env = data['env_feature']
        nbr_motion = data['neighbor_motion']
                
        batch_size = nbr_motion.shape[0]
        agent_num = nbr_motion.shape[1]
        
        ## encode ego
        ego_input_seq = torch.cat([ego_motion, ego_feature], axis=2)
        ego_enc = self.input_ego_fc(ego_input_seq) # .permute(1, 0, 2) # [B, L, D] -> [L, B, D]
        ego_enc = self.pos_enc_ego(ego_enc)

        ## encode environment
        env_enc = self.input_env_fc(input_env) # [B, L, D]
        env_enc = self.pos_enc_env(env_enc)
        
        
        ## encode neighbor
        nbr_motion = torch.permute(nbr_motion, (2, 1, 0, 3)).reshape(-1, batch_size, nbr_motion.shape[-1]) # [L*N, B, f]
        tf_in = self.input_nbr_fc(nbr_motion.view(-1, nbr_motion.shape[-1])).view(-1, batch_size, self.model_dim)  # [N*his_len, B, D]
        agent_enc_shuffle = None if self.agent_enc_shuffle else None
        tf_in_pos = self.agent_pos_encoder(tf_in, num_a=agent_num, agent_enc_shuffle=agent_enc_shuffle)
        
        ## calculate source mask with varied agent_num
        batch_mask = data['neighbor_mask'].transpose(1, 2).reshape(batch_size, -1)
        src_mask = torch.matmul(batch_mask.unsqueeze(2), batch_mask.unsqueeze(1)) # [B, 72, 72]
        src_mask = src_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).reshape(-1, self.input_length * agent_num, self.input_length * agent_num)
        src_mask += -1
        src_mask[src_mask < -0.1] = -1e4
        

        pos_mask_in = self.cal_mask_input(data)
        nbr_input = tf_in_pos # [L*N, B, D]
        
        # 1st level
        context_nbrs = self.af_encoder(nbr_input, mask=src_mask, num_agent=agent_num, pos_mask=pos_mask_in)
        nbr_enc_1 = self.context_fc(context_nbrs).view(-1, batch_size, self.model_dim).permute(1, 0, 2) # [B, LN, D]
        env_enc_1 = self.tf_encoder_env(env_enc)
        ego_enc_1 = self.tf_encoder_ego(ego_enc)
        
        # 2nd level
        env_enc_2, nbr_2_1 = self.HCA_env_nbr(env_enc_1, nbr_enc_1)
        ego_enc_2, nbr_2_2 = self.HCA_ego_nbr(ego_enc_1, nbr_enc_1)
        
        # 3rd level
        out_env_3, _ = self.decoder_env_MHA(env_enc_2, nbr_2_1, nbr_2_1)
        out_ego_3, _ = self.decoder_ego_MHA(ego_enc_2, nbr_2_2, nbr_2_2)
        
        out = out_env_3 + out_ego_3
        
        # decoder
        ret_dict = self.intention2traj_decoder(out, data)   

        return ret_dict

    