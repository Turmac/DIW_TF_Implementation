import numpy as np


N_THRESHOLD = 15

def reset_record():
    record = {
        'n_threshold': N_THRESHOLD,
        'eq_correct_count': np.zeros(N_THRESHOLD),
        'not_eq_correct_count': np.zeros(N_THRESHOLD),
        'eq_count': 0,
        'not_eq_count': 0,
        'threshold': 0.1*np.array(range(N_THRESHOLD)),
        'WKDR': np.zeros((N_THRESHOLD, 4))
    }
    return record


def _classify(zA, zB, gt, threshold):
    if zA - zB > threshold:
        res = 1
    elif zA - zB < -threshold:
        res = -1
    else:
        res = 0
    return (res == gt)


def _count_correct(output, target, record):
    for idx in range(len(target)):
        yA = target[idx][0]
        xA = target[idx][1]
        yB = target[idx][2]
        xB = target[idx][3]

        zA = output[yA][xA]
        zB = output[yB][xB]

        gt = target[idx][4]

        for tau_idx in range(record['n_threshold']):
            if _classify(zA, zB, gt, record['threshold'][tau_idx]):
                if gt == 0:
                    record['eq_correct_count'][tau_idx] += 1
                else:
                    record['not_eq_correct_count'][tau_idx] += 1
        
        if gt == 0:
            record['eq_count'] += 1
        else:
            record['not_eq_count'] += 1


def evaluate(output, target):
    """
    Args:
        Output: np array, N x height x width x 1
        target: np array, N x 800 x 5
    """
    record = reset_record()

    n_iters = output.shape[0]
    for i in range(n_iters):
        _count_correct(output[i], target[i], record)
    
    max_min = 0
    max_min_k = 1
    for tau_idx in range(record['n_threshold']):
        record['WKDR'][tau_idx][0] = record['threshold'][tau_idx]
        record['WKDR'][tau_idx][1] = float(record['eq_correct_count'][tau_idx]+record['not_eq_correct_count'][tau_idx])/float(record['eq_count']+record['not_eq_count'])
        record['WKDR'][tau_idx][2] = float(record['eq_correct_count'][tau_idx])/float(record['eq_count'])
        record['WKDR'][tau_idx][3] = float(record['not_eq_correct_count'][tau_idx])/float(record['not_eq_count'])
    
        if min(record['WKDR'][tau_idx][2], record['WKDR'][tau_idx][3]) > max_min:
            max_min = min(record['WKDR'][tau_idx][2], record['WKDR'][tau_idx][3])
            max_min_k = tau_idx
    
    return 1 - max_min
