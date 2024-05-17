import argparse
import os

__all__ = ['parse_base_args']

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--loss_scale', default=1.0, type=float)
    parser.add_argument('--loss_reduction', default='mean', type=str)
    parser.add_argument('--mode', default='train', type=str)

    parser.add_argument('--model', default='rlr_predictor', type=str)
    parser.add_argument('--data_root', default='data/training_data_pickle_3s', type=str)

    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--split_shuffle', default=True, type=bool)
    parser.add_argument('--test_split', default=0.4, type=float)

    parser.add_argument('--lr', default=5e-04, type=float) 
    parser.add_argument('--input_length', default=12, type=int)
    parser.add_argument('--output_length', default=36, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    
    parser.add_argument('--pred_GMM', default=True, type=bool)
    parser.add_argument('--pred_dim', default=2, type=int)
    parser.add_argument('--pred_GMM_dim', default=2, type=int)
    parser.add_argument('--pred_intention_dim', default=2, type=int)
    parser.add_argument('--input_dim', default=20, type=int)
    parser.add_argument('--input_env_dim', default=11, type=int)
    parser.add_argument('--input_ego_dim', default=9, type=int)
    parser.add_argument('--input_ego_map_dim', default=5, type=int)
    parser.add_argument('--input_nbr_dim', default=4, type=int)
    parser.add_argument('--K', default=6, type=int)

    # Transformer
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_layers', default=8, type=int)
    parser.add_argument('--dim_ff', default=64, type=int)

    # AgentFormer
    parser.add_argument('--model_dim', default=64, type=int)
    parser.add_argument('--agent_enc_shuffle', default=False, type=bool)
    parser.add_argument('--pos_concat', default=False, type=bool)
    parser.add_argument('--max_agent_len', default=32, type=int)
    parser.add_argument('--use_agent_enc', default=False, type=bool)
    parser.add_argument('--agent_enc_learn', default=False, type=bool)

    # RLR args
    parser.add_argument('--rlr_delta_t', default=0.4, type=float)
    parser.add_argument('--rlr_loss_weight', default=False, type=bool)

    # Save model
    parser.add_argument('--model_save_freq', default=1, type=int)
    parser.add_argument('--results_path', default='results/', type=str)
    parser.add_argument('--results_dir', default=None, type=str)

    return parser.parse_known_args()[0]