import sys
import os
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from configs.base_configs import parse_args
from utils.models.model_libs import build_model
from utils.data.data_utils import set_seed, build_data_loader
from utils.eval.eval_utils import evaluate_results, evaluate_trajectory_prediction, evaluate_consistency
from utils.loss.loss import TrajLoss, get_batch_best_mode
from utils.common_utils import makedir


def train(args, model, epoch, train_loader, device, optimizer, criterion_list, verbose=True):
    running_loss = 0.0
    intention_running_loss = 0.0
    trajectory_running_loss = 0.0
    verbose_num = 50

    model.train()
    for batch_idx, data in enumerate(train_loader, 0):

        for k in data.keys():
            data[k] = data[k].type(torch.FloatTensor).to(device)
        data['ego_is_rlr'] = data['ego_is_rlr'].type(torch.long).to(device)
        data['ego_is_rlr_step'] = data['ego_is_rlr_step'].type(torch.long).to(device)
        
        fut_traj = data['ego_motion_y']
        fut_is_rlr = data['ego_is_rlr']
        fut_is_rlr_step = data['ego_is_rlr_step']
        
        optimizer.zero_grad() 

        ret_dict = model(data)
        out_fut_is_rlr = ret_dict['pre_is_rlr_whole'] 
        out_fut_is_rlr_step = ret_dict['pre_is_rlr_step']
        out_fut_traj = ret_dict['pre_motion']
        out_fut_traj_probas = ret_dict['pre_motion_probas']

        best_mode = get_batch_best_mode(fut_traj, out_fut_traj).cpu()
        out_fut_is_rlr = out_fut_is_rlr[torch.arange(best_mode.size(0)), best_mode]
        out_fut_is_rlr_step = out_fut_is_rlr_step[torch.arange(best_mode.size(0)), best_mode]

        # Prepare for CELoss calculation
        out_fut_is_rlr = out_fut_is_rlr.type(torch.FloatTensor).to(device)
        out_fut_is_rlr_step = out_fut_is_rlr_step.type(torch.FloatTensor).to(device)        
        out_fut_is_rlr_step = torch.permute(out_fut_is_rlr_step, (0, 2, 1))

        fut_is_rlr = F.one_hot(fut_is_rlr, num_classes=2).type(torch.float)
        fut_is_rlr_step = F.one_hot(fut_is_rlr_step, num_classes=2).type(torch.float) 
        fut_is_rlr_step = torch.permute(fut_is_rlr_step, (0, 2, 1))
        
        # Calculate loss
        intention_criterion, traj_criterion = criterion_list
        intention_loss = intention_criterion(out_fut_is_rlr, fut_is_rlr) + intention_criterion(out_fut_is_rlr_step, fut_is_rlr_step) 
        traj_loss = traj_criterion(fut_traj, out_fut_traj, out_fut_traj_probas)
        loss = args.loss_scale * intention_loss + traj_loss 

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        intention_running_loss += intention_loss.item()
        trajectory_running_loss += traj_loss.item()
            
        if batch_idx % verbose_num == 0 and verbose:
            print('[epoch %d, batch %1d] loss: %.3f, intention loss: %.3f, trajectory loss: %.3f, ' % (
                epoch + 1, 
                batch_idx+1, 
                running_loss / verbose_num,
                intention_running_loss / verbose_num,
                trajectory_running_loss / verbose_num,
            ))
            running_loss = 0.0
            intention_running_loss = 0.0
            trajectory_running_loss = 0.0
        
            
def test(args, model, epoch, test_loader, device, criterion_list, verbose=True):
    verbose_num = 50
    running_loss = 0

    ret_traj = {}
    model.eval()
    with torch.no_grad(): 
        for batch_idx, data in enumerate(test_loader, 0):
            for k in data.keys():
                data[k] = data[k].type(torch.FloatTensor).to(device)
            data['ego_is_rlr'] = data['ego_is_rlr'].type(torch.long).to(device)
            data['ego_is_rlr_step'] = data['ego_is_rlr_step'].type(torch.long).to(device)

            his_traj = data['ego_motion_x']
            fut_traj = data['ego_motion_y']
            fut_is_rlr = data['ego_is_rlr']
            fut_is_rlr_step = data['ego_is_rlr_step']

            ret_dict = model(data)
            out_fut_is_rlr = ret_dict['pre_is_rlr_whole']
            out_fut_is_rlr_step = ret_dict['pre_is_rlr_step']
            out_fut_traj = ret_dict['pre_motion']
            out_fut_traj_probas = ret_dict['pre_motion_probas']
            
            best_mode = get_batch_best_mode(fut_traj, out_fut_traj).cpu()
            out_fut_is_rlr = out_fut_is_rlr[torch.arange(best_mode.size(0)), best_mode]
            out_fut_is_rlr_step = out_fut_is_rlr_step[torch.arange(best_mode.size(0)), best_mode]
            out_fut_is_rlr = out_fut_is_rlr.type(torch.FloatTensor).to(device)
            out_fut_is_rlr_step = out_fut_is_rlr_step.type(torch.FloatTensor).to(device)
            out_fut_is_rlr_step_CE = torch.permute(out_fut_is_rlr_step, (0, 2, 1))
            fut_is_rlr_onehot = F.one_hot(fut_is_rlr, num_classes=2).type(torch.float)
            fut_is_rlr_step_CE = F.one_hot(fut_is_rlr_step, num_classes=2).type(torch.float)
            fut_is_rlr_step_CE = torch.permute(fut_is_rlr_step_CE, (0, 2, 1))

            # Loss
            intention_criterion, traj_criterion = criterion_list
            intention_loss = intention_criterion(out_fut_is_rlr, fut_is_rlr) + intention_criterion(out_fut_is_rlr_step_CE, fut_is_rlr_step_CE) 
            traj_loss = traj_criterion(fut_traj, out_fut_traj, out_fut_traj_probas)
            loss = args.loss_scale * intention_loss + traj_loss
            
            running_loss += loss.item()
            if batch_idx % verbose_num == 0 and verbose:
                print('TEST: [epoch %d, batch %1d] loss: %.3f' % (epoch + 1, batch_idx+1, running_loss / verbose_num))
                running_loss = 0.0
            
            if batch_idx == 0:
                
                ret_traj['his_traj'] = his_traj.cpu()
                ret_traj['fut_traj_org'] = fut_traj.cpu()
                ret_traj['fut_traj_pre_k'] = out_fut_traj.cpu()
                ret_traj['fut_traj_pre'] = out_fut_traj[torch.arange(best_mode.size(0)), best_mode].cpu()
                ret_traj['fut_is_rlr_org'] = fut_is_rlr.cpu()
                ret_traj['fut_is_rlr_pre'] = out_fut_is_rlr.cpu()
                ret_traj['fut_is_rlr_step_org'] = fut_is_rlr_step.cpu()
                ret_traj['fut_is_rlr_step_pre'] = out_fut_is_rlr_step.cpu()
                ret_traj['intersection_polygon'] = data['intersection_polygon'].cpu()
                ret_traj['m_inverse'] = data['m_inverse'].cpu()
                ret_traj['anchor_point'] = data['anchor_point'].cpu()
                ret_traj['id_int'] = data['id_int'].cpu()
                ret_traj['ttrlr'] = data['ego_ttrlr'].cpu()
                
            ret_traj['fut_traj_pre_k'] = np.concatenate([ret_traj['fut_traj_pre_k'], out_fut_traj.cpu()], axis=0)
            ret_traj['fut_traj_org'] = np.concatenate([ret_traj['fut_traj_org'], fut_traj.cpu()], axis=0)
            ret_traj['fut_traj_pre'] = np.concatenate([ret_traj['fut_traj_pre'], out_fut_traj[torch.arange(best_mode.size(0)), best_mode].cpu()], axis=0)
            ret_traj['his_traj'] = np.concatenate([ret_traj['his_traj'], his_traj.cpu()], axis=0)
            ret_traj['fut_is_rlr_pre'] = np.concatenate([ret_traj['fut_is_rlr_pre'], out_fut_is_rlr.cpu()], axis=0)
            ret_traj['fut_is_rlr_org'] = np.concatenate([ret_traj['fut_is_rlr_org'], fut_is_rlr.cpu()], axis=0)
            ret_traj['fut_is_rlr_step_pre'] = np.concatenate([ret_traj['fut_is_rlr_step_pre'], out_fut_is_rlr_step.cpu()], axis=0)
            ret_traj['fut_is_rlr_step_org'] = np.concatenate([ret_traj['fut_is_rlr_step_org'], fut_is_rlr_step.cpu()], axis=0)
            ret_traj['intersection_polygon'] = np.concatenate([ret_traj['intersection_polygon'], data['intersection_polygon'].cpu()], axis=0)
            ret_traj['m_inverse'] = np.concatenate([ret_traj['m_inverse'], data['m_inverse'].cpu()], axis=0)
            ret_traj['anchor_point'] = np.concatenate([ret_traj['anchor_point'], data['anchor_point'].cpu()], axis=0)
            ret_traj['id_int'] = np.concatenate([ret_traj['id_int'], data['id_int'].cpu()], axis=0)
            ret_traj['ttrlr'] = np.concatenate([ret_traj['ttrlr'], data['ego_ttrlr'].cpu()], axis=0)

    return ret_traj


def main(args):
    
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print("torch version : {}".format(torch.__version__))
    print("cudnn version : {}".format(torch.backends.cudnn.version()))
    
    
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    set_seed(int(args.seed))
    
    model_results_path = makedir(args)
    model = build_model(args).to(device)

    # loss
    reduction = args.loss_reduction
    intention_criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
    traj_criterion = TrajLoss()
    criterion_list = [intention_criterion, traj_criterion]
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # load data
    train_loader = build_data_loader(args, mode='train')
    test_loader = build_data_loader(args, mode='test')
    
    num_epoch = args.epochs
    if args.mode == 'train':
        start_epoch = 0
        
        # load pre-trained model if start_epoch > 0
        if args.start_epoch > 0:
            start_epoch = args.start_epoch
            cp_path = model_results_path % start_epoch
            print(f'loading model from checkpoint: {cp_path}')
            model_cp = torch.load(cp_path, map_location='cpu')
            model.load_state_dict(model_cp['model_dict'])
        
        # training
        for epoch in range(start_epoch, num_epoch + start_epoch):
            train(args, model, epoch, train_loader, device, optimizer, criterion_list)
            ret_traj = test(args, model, epoch, test_loader, device, criterion_list)
            
            evaluate_trajectory_prediction(ret_traj)
            evaluate_results(ret_traj)
            evaluate_consistency(ret_traj)

            if args.model_save_freq > 0 and (epoch + 1) % args.model_save_freq == 0:
                cp_path = model_results_path % (epoch + 1)
                model_cp = {
                    'model_dict': model.state_dict(), 
                    'opt_dict': optimizer.state_dict(), 
                    'scheduler_dict': scheduler.state_dict(), 
                    'epoch': epoch + 1}
                torch.save(model_cp, cp_path)
            
            scheduler.step()
            
    elif args.mode == 'test':
        epoch = 30

        cp_path =  model_results_path % (epoch)
        print(f'loading model from checkpoint: {cp_path}')
        model_cp = torch.load(cp_path, map_location='cpu')
            
        model.load_state_dict(model_cp['model_dict'])
        ret_traj = test(args, model, epoch-1, test_loader, device, criterion_list)
            
        evaluate_long_results(ret_traj, epoch, name=result_name)
        evaluate_trajectory_prediction(ret_traj)
        evaluate_results(ret_traj)
        _, rlr_rule = evaluate_consistency(ret_traj)
        
        
if __name__ == '__main__':
    main(parse_args())