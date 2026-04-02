import torch.nn as nn

from .utils import *

import torch.optim as optim

def first(l, _):
    return l[0]

model_configs = {

    'ID_grid0_GDP':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1410)], 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]#,
            #[
            #    [nn.Linear(1410, 1)]
            #]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True, True],
                        3:[True],
                        #4:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first,
                            #4:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1)), ('special_task_GDP', slice(1, 2))],
                            #4:[('side_tasks', slice(0, 1))]
                            }
    },
    'ID_grid0_GDP_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1410)], 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]#,
            #[
            #    [nn.Linear(1410, 1)]
            #]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True, True],
                        3:[True],
                        #4:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first,
                            #4:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1)), ('special_task_GDP', slice(1, 2))],
                            #4:[('side_tasks', slice(0, 1))]
                            }
    },

    'ID_grid0':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid1':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.2), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid2':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.4), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid3':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.6), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid4':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.75), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid4_boosted':{
        'basemodel':None,
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.75), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid0_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.1), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.1), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.1), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid1_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.2), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.2), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.2), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid2_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.4), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.4), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.4), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid3_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.6), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.6), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.6), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    },
    'ID_grid4_full':{
        'basemodel':"ViT-SO400M-14-SigLIP-384",
        'geolocation_model_extension':[
            [
                [nn.Dropout(p=0.75), nn.Linear(1152, 2312)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)],
                [nn.Dropout(p=0.75), nn.Linear(1152, 1)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(2312+17, 11171)]
            ],
            [ 
                [nn.ReLU(), nn.Dropout(p=0.75), nn.Linear(11171, 1410)]
            ],
            [
                [nn.Softmax(dim=1)]
            ]
        ],
    'unfreeze_basemodel_params_conf':slice(0, 0),
    'preprocess':None,
    'optimizer':optim.AdamW,
    'optimizer_params':{'lr':0.0001},
    'target_outputs':{
                        0:[False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                        1:[False],
                        2:[True],
                        3:[True]
                    },
    'concurrent_reduction':{
                            0:torch.cat,
                            1:first,
                            2:first,
                            3:first
                            },
    'tasks':{
                            0:[('side_tasks', slice(0, 16))],
                            2:[('geolocation', slice(0, 1))]
                            }
    }
}

system_configs = {
    'SYS_ID_OSV5M_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4_boosted_full',
        'predefined_region_grid':'ID1',
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':96,
        'data_location':'datasets/osv5m/copied',
        'panorama':False
    },
    'SYS_ID_OSV5M':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4_boosted',
        'predefined_region_grid':'ID1',
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_osv5m',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'datasets/osv5m/copied',
        'panorama':False
    },
    'SYS_ID_OSV5M_noboost':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4',
        'predefined_region_grid':'ID1',
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_osv5m',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':96,
        'data_location':'datasets/osv5m/copied',
        'panorama':False
    },
    'SYS_ID_grid00':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4_full',
        'predefined_region_grid':'ID1',
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':96,
        'data_location':'datasets/osv5m/copied',
        'panorama':False
    },

        'SYS_ID_grid01':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0_full',
        'predefined_region_grid':'ID1',
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':96,
        'data_location':'datasets/osv5m/copied',
        'panorama':False
    },

    'SYS_ID_grid0_GDP':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss(), nn.MSELoss()]#,
                        #4:[nn.MSELoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
                        2:[0.16374254098546728, 100]#,
                        #4:[100]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0_GDP',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation', 'special_task_GDP'],
            4:['GDP_better']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
    },

    'SYS_ID_grid0':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
    },
    'SYS_ID_grid1':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid1',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
    },
    'SYS_ID_grid2':{
            "auxiliary_loss":{
                            0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                            2:[nn.CrossEntropyLoss()]
                            },
            "loss_multiplier":{
                            0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            2:[0.16374254098546728]
                            },
            "tau":33.58948578827294,
            'COUNTRIES_T':None,
            'blur_system':None,
            'save_system':True,
            'model_ID':'ID_grid2',
            'predefined_region_grid':None,
            'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
            'variable_names':{
                0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                    'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                    'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                    'precipitation 6 June', 'GDP'],
                2:['geolocation']
            },
            'batch_size':1024,
            'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid3':{
            "auxiliary_loss":{
                            0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                            2:[nn.CrossEntropyLoss()]
                            },
            "loss_multiplier":{
                            0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            2:[0.16374254098546728]
                            },
            "tau":33.58948578827294,
            'COUNTRIES_T':None,
            'blur_system':None,
            'save_system':True,
            'model_ID':'ID_grid3',
            'predefined_region_grid':None,
            'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
            'variable_names':{
                0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                    'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                    'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                    'precipitation 6 June', 'GDP'],
                2:['geolocation']
            },
            'batch_size':1024,
            'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid4':{
            "auxiliary_loss":{
                            0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                            nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                            2:[nn.CrossEntropyLoss()]
                            },
            "loss_multiplier":{
                            0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            2:[0.16374254098546728]
                            },
            "tau":33.58948578827294,
            'COUNTRIES_T':None,
            'blur_system':None,
            'save_system':True,
            'model_ID':'ID_grid4',
            'predefined_region_grid':None,
            'on_embeddings':'ViT-SO400M-14-SigLIP-384_mix',
            'variable_names':{
                0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                    'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                    'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                    'precipitation 6 June', 'GDP'],
                2:['geolocation']
            },
            'batch_size':1024,
            'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid0_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid1_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid1',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid2_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid2',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid3_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid3',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
        },
    'SYS_ID_grid4_clean':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()],
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4',
        'predefined_region_grid':None,
        'on_embeddings':'ViT-SO400M-14-SigLIP-384',
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':1024,
        'data_location':'storage',
        'panorama':True
        },

    'SYS_ID_grid0_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid0_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':24,
        'data_location':'storage',
        'panorama':True
        },

    'SYS_ID_grid1_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid1_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':24,
        'data_location':'storage',
        'panorama':True
        },

    'SYS_ID_grid2_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid2_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':24,
        'data_location':'storage',
        'panorama':True
        },

    'SYS_ID_grid3_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid3_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':24,
        'data_location':'storage',
        'panorama':True
        },

    'SYS_ID_grid4_full':{
        "auxiliary_loss":{
                        0:[nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(), nn.MSELoss(),
                           nn.MSELoss(), nn.MSELoss(), nn.MSELoss()], 
                        2:[nn.CrossEntropyLoss()]
                        },
        "loss_multiplier":{
                        0:[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        2:[0.16374254098546728]
                        },
        "tau":33.58948578827294,
        'COUNTRIES_T':None,
        'blur_system':None,
        'save_system':True,
        'model_ID':'ID_grid4_full',
        'predefined_region_grid':None,
        'on_embeddings':None,
        'variable_names':{
            0:['precipitation Exact', 'mean temperature 6 Exact', 'solar radiation Exact', 'minimum temperature Exact', 'maximum temperature Exact',
                'wind speed Exact', 'water vapour pressure Exact', 'precipitation 6 Exact', 'precipitation June',
                'mean temperature 6 June', 'solar radiation June', 'minimum temperature June', 'maximum temperature June', 'wind speed June', 'water vapour pressure June',
                'precipitation 6 June', 'GDP'],
            2:['geolocation']
        },
        'batch_size':24,
        'data_location':'storage',
        'panorama':True
        }
}
