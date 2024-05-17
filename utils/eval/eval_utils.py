import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from utils.common_utils import is_in_poly_01

def evaluate_binary_classification(labels, predictions):
    cm = confusion_matrix(labels, predictions)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    results = {
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    return results


def evaluate_results(ret_traj):

    fut_is_rlr_pre = [1 if x[1] > x[0] else 0 for x in ret_traj['fut_is_rlr_pre']]

    evaluation_results = evaluate_binary_classification(ret_traj['fut_is_rlr_org'], fut_is_rlr_pre)
    print('Evaluation Results:')
    for metric, value in evaluation_results.items():
        if metric == 'Confusion Matrix':
            print('{}: \n{}'.format(metric, value))
        else:
            print('{}: {:.4f}'.format(metric, value))

    return evaluation_results


def evaluate_trajectory_prediction(ret_traj):
    
    true_trajectories = ret_traj['fut_traj_org']
    predicted_trajectories = ret_traj['fut_traj_pre']

    fde = np.mean(np.linalg.norm(true_trajectories[:, -1] - predicted_trajectories[:, -1, 0:2], axis=1))
    ade = np.mean(np.mean(np.linalg.norm(true_trajectories - predicted_trajectories[:, :, 0:2], axis=2), axis=1))
    
    print('FDE: {:.4f}'.format(fde))
    print('ADE: {:.4f}'.format(ade))
    
    return fde, ade

def evaluate_consistency(ret_traj):
    
    his_true_trajectories = ret_traj['his_traj']
    predicted_trajectories = ret_traj['fut_traj_pre']
    
    fut_is_rlr_pre = np.array([1 if x[1] > x[0] else 0 for x in ret_traj['fut_is_rlr_pre']])

    last_pnt = predicted_trajectories[:, -1, :] # [B, 2]
    first_pnt = his_true_trajectories[:, -1, 0:2] # [B, 2]
    traj_pnt = last_pnt - first_pnt
    rlr_intention_rule = np.zeros(len(traj_pnt))

    for i in range(len(traj_pnt)):
        x, y = traj_pnt[i]
        poly = ret_traj['intersection_polygon'][i]
        rlr_intention_rule[i] = is_in_poly_01(x, y, [poly]) # [B] 0/1
    
    consistency = np.zeros(len(rlr_intention_rule))
    consistency[rlr_intention_rule == fut_is_rlr_pre] = 1.0
    consistency_score = consistency.mean()
    
    print('Consistency score: {:.4f}'.format(consistency_score))
    
    evaluation_results = evaluate_binary_classification(ret_traj['fut_is_rlr_org'], rlr_intention_rule)
    
    print('Rule-based Evaluation Results:')
    for metric, value in evaluation_results.items():
        if metric == 'Confusion Matrix':
            print('{}: \n{}'.format(metric, value))
        else:
            print('{}: {:.4f}'.format(metric, value))

    return consistency, rlr_intention_rule

