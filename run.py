from sklearn.metrics import root_mean_squared_error, r2_score
from bayes_opt import BayesianOptimization
import gc
from torch.autograd import Variable
import torchvision.transforms as T

from GeoGuesserSystem import *

import torch
import json
import pickle
import os
import gc

if __name__ == '__main__':
    
    def first(l, _):
        return l[0]

    train_dataset, test_dataset, pct_n, mapping, shp_n, pct, shp, countries = process_data(None, 'SYS_ID_grid0', 'ViT-SO400M-14-SigLIP-384_mix', None)

    def FineTuneGlobalIterator(loss_multiplier, tau, layer1, layer2, drp):

        system_conf = system_configs['SYS_ID_grid0']
        system_conf["loss_multiplier"][2][0] = loss_multiplier

        system_conf['tau'] = tau

        layer1 = int(layer1)
        layer2 = int(layer2)

        extension = [
                [
                    [nn.Dropout(p=drp), nn.Linear(1152, layer1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)],
                    [nn.Dropout(p=drp), nn.Linear(1152, 1)]
                ],
                [ 
                    [nn.ReLU(), nn.Dropout(p=drp), nn.Linear(layer1+17, layer2)]
                ],
                [ 
                    [nn.ReLU(), nn.Dropout(p=drp), nn.Linear(layer2, 1410)]
                ],
                [
                    [nn.Softmax(dim=1)]
                ]
            ]

        model = GeoBrainNetwork(None, None, extension,
                                {
                                    0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                                    1:[False],
                                    2:[True],
                                    3:[True]
                                },
                                {
                                    0:torch.cat,
                                    1:first,
                                    2:first,
                                    3:first
                                },
                                {
                                    0:[('side_tasks', slice(0, 16))],
                                    2:[('geolocation', slice(0, 1))]
                                }).to('cuda')
        
        optimizer = optim.AdamW(model.parameters(), **{'lr':0.0001})

        id = np.random.choice(np.arange(10000, 100000))

        if os.path.isfile('hypertunings/config_%i.json'%id):
            id = np.random.choice(np.arange(10000, 100000))

        BR = BRAIN()
        BR.NN = model.to(DEVICE)
        BR.train_dataset = train_dataset
        BR.test_dataset = test_dataset
        BR.loss = system_conf['auxiliary_loss']
        BR.loss_multiplier = system_conf['loss_multiplier']
        BR.tau = system_conf['tau']
        BR.pct_n = pct_n
        BR.pct = pct
        BR.mapping = mapping
        BR.shp_n = shp_n
        BR.shp = shp
        BR.device = DEVICE
        BR.optimizer = optimizer
        BR.y_variable_names = system_conf['variable_names']
        BR.batch_size=system_conf['batch_size']

        BR.prepare_system(list(countries))

        BR.prepare_dataloaders()
        BR.train(40, name='hypertunings/Version %i' % id)
        BR.generate_test_main(on='test')
        
        acc = {}
        for j in [2, 4, 6, 12]:
            t = BR.task_summary_test['geolocation'].map(lambda x: x[:j]).groupby(['pred'])['real'].value_counts().unstack(-1)
            acc[j] = np.nansum(np.diag(t))/np.nansum(t)

        y = BR.task_summary_test.drop(columns='geolocation')
        SCORE = 0
        for c in y.columns.get_level_values(0).unique():
            SCORE-=(y[c].diff(axis=1)['real']**2).mean()
        SCORE += acc[4]*20

        params = {
            'loss_multiplier':float(loss_multiplier), 
            'tau':float(tau), 
            'layer1':int(layer1), 
            'layer2':int(layer2), 
            'drp':float(drp),
            'SCORE':float(SCORE),
            'id':int(id)
        }

        with open('hypertunings/config_%i.json'%id, 'w', encoding='utf-8') as f:
            json.dump(params, f)

        BR.task_summary_test.to_csv('hypertunings/df_%i.csv' % id)

        del BR
        gc.collect()
        torch.cuda.empty_cache()

        return SCORE

    with open('hypertunings/optim_obj.pkl', 'rb') as f:
        optimizer = pickle.load(f)

    #optimizer = BayesianOptimization(FineTuneGlobalIterator, pbounds = {'loss_multiplier':(0.01, 1), 
    #                                                        'tau':(10, 300), 
    #                                                        'layer1':(500, 3500), 
    #                                                        'layer2':(500, 15000),
    #                                                        'drp':(0, 0.7)})
    
    optimizer.maximize(
        init_points=10,
        n_iter=200,
    )
    with open('hypertunings/optim_obj.pkl', 'wb') as f:
        pickle.dump(optimizer, f, pickle.HIGHEST_PROTOCOL)