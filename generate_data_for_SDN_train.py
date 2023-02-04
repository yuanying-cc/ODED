import torch
import numpy as np
import re
import os



if __name__ == '__main__':


    path = 'data/suboptimal_demos'

    data_names = os.listdir(path)
    data_names_true = [name for name in data_names if re.split('\.', name)[-1]=='pt']
    keys = [int(re.split('_|\.', s)[-2]) for s in data_names_true]

    names = dict(zip(keys, data_names_true))
    names2 = dict(zip(data_names_true, keys))


    RESULT_DIR = os.path.join('data', '.'.join(__file__.split('.')[:-1]))
    os.makedirs(RESULT_DIR, exist_ok=True)
    token = 'concat_train_test_data_for_SDN_training'
    test_path = os.path.join(RESULT_DIR, token)
    os.makedirs(test_path, exist_ok=True)




    sorted_keys = sorted(keys)

    sorted_names = [names[key] for key in sorted_keys]
    half_length = 50

    train_xy_dict = {} 

    for name in sorted_names:
        data = torch.load(os.path.join(path, name))
        data_sa = torch.cat([data['states'], data['actions']],2)
        train_xy_dict['{}'.format(names2[name])] = data_sa.reshape(data_sa.shape[0], -1)


    cat_train_xy_dict = {}
    for name1,name2 in zip(list(train_xy_dict.keys())[:-1], list(train_xy_dict.keys())[1:]):

        new_name = name1+'_'+name2
        print(new_name)
        tmp = torch.cat([train_xy_dict[name1][half_length:,:], train_xy_dict[name2][:half_length,:]], 1)
        print(tmp.shape)
        cat_train_xy_dict[new_name] = tmp
                

    torch.save(cat_train_xy_dict, os.path.join(test_path,  'train_test_xy.pt'))


    train_test_xy_np = torch.cat([xy for xy in cat_train_xy_dict.values()])

    np.savetxt(os.path.join(test_path, 'train_test_xy.csv'), train_test_xy_np, delimiter=',')

