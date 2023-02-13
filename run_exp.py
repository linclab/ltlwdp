import sys
import os
import pdb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ariamis.base_models import DenseLayer, DenseNet, build_densenet
from ariamis.base_models import LayerNormLayer, BatchNormLayer
from ariamis.base_models import set_attr_for_all_layers
from ariamis.ei_models import EiDenseWithShunt
from ariamis.ei_models import DalesANN_cSGD_UpdatePolicy, DalesANN_SGD_UpdatePolicy, DalesANN_UpdatePolicy
from ariamis.ei_models import EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy
from ariamis.ngd import FisherCalculator, FisherDenseNet, FisherLayerMixin
from ariamis.column_ei_models import ColummEiDense, ColumnEiSGD
from ariamis.column_ei_models import ColumnEiWeightInitPolicy, PostiveWeightInitPolicy

import ariamis.utils.all_utils as utils
from ariamis.data import mnist
from ariamis.train import acc_func
from ariamis.utils.all_utils import Progbar

from orion.client import report_results as orion_report_results

def in_notebook():
    for arg in sys.argv:
        if "jupyter" in arg:
            return True
    return False

# -------------------------------------------
#          Define Train Loop
# -------------------------------------------
def main_loop(model, train_dataloader, val_x, val_y, flags, device, save_model=False, validation_interval=1):
    """
    Args:
        - model
        - train_dataloader.dataset is expected to be on device
        - val_x, val_y are the full validation set
        - flags: object to hold param options
        - save_model: bool, if true saves model at the end of the training (no early stopping)
        - validation_interval: check full validation set when update_i % validation_interval == 0
    """
    update_i = 0   # update counter
    total_steps = flags.n_epochs*len(train_dataloader)
    loss_func = F.cross_entropy

    #                               -------------  Build results dict -----------------

    step_res_data = [] # "step_res_data" is logged when update_i % validation_interval == 0
    step_res_data += ['x_update'] # a list of update indices
    step_res_data += ['test_loss', 'train_loss']
    step_res_data += ['test_err',  'train_err']
    step_res_data += ['test_acc',  'train_acc']

    all_epoch_results= {k:[] for k in step_res_data}

    epoch_resolution  = ['x_epoch', 'train_loss_epoch', 'test_loss_epoch']
    epoch_resolution += ['test_err_epoch', 'train_err_epoch', 'test_acc_epoch', 'train_acc_epoch', ]

    all_epoch_results.update({k:[] for k in epoch_resolution})

    print("Keys to save in results are :", all_epoch_results.keys())


     #                                ---------- Training Loop -----------------

    for epoch_i in range(flags.n_epochs):

        if in_notebook(): verbose_code = 1
        else: verbose_code = 1  # if in terminal can pass 2 to only print output at the end of an epoch
        prog_bar = Progbar(len(train_dataloader), verbose=verbose_code)

        model.train()

        #----- Train on epoch  -----------------
        for batch_i, (x, y) in enumerate(train_dataloader):
            yhat = model(x)
            acc = acc_func(yhat,y)
            err = (1 - acc)*100
            loss = loss_func(yhat,y)
            loss.backward(retain_graph=True)

            all_epoch_results['train_loss'].append(loss.item())
            all_epoch_results['train_acc'].append(acc)
            all_epoch_results['train_err'].append(err)

            if batch_i%20 ==0:
                prog_bar.update(batch_i, values= [('err', all_epoch_results['train_err'][-1]),
                                                  ('loss',all_epoch_results['train_loss'][-1])])

            if type(model) is DenseNet:
                model.update(lr=flags.lr, flags=flags)

            elif type(model) is FisherDenseNet:
                model.update(lr=flags.lr, input_batch=x.view(-1, model.n_input))

            model.zero_grad()
            update_i += 1

            # ------ Validation  ------------
            if update_i % validation_interval==0:
                model.eval()
                with torch.no_grad():
                    val_yhat = model(eval_x)
                    val_loss = F.cross_entropy(val_yhat, eval_y)
                    val_acc = acc_func(val_yhat, eval_y)
                    val_err = (1 - val_acc)*100
                    all_epoch_results['test_loss'].append(val_loss.item())
                    all_epoch_results['test_err'].append(val_err)
                    all_epoch_results['test_acc'].append(val_acc)
                    all_epoch_results['x_update'].append(update_i)
                model.train()

        # epoch resolution results
        all_epoch_results['x_epoch'].append(update_i-1)
        all_epoch_results['test_err_epoch'].append(all_epoch_results['test_err'][-1])
        all_epoch_results['test_loss_epoch'].append(all_epoch_results['test_loss'][-1])
        all_epoch_results['test_acc_epoch'].append(all_epoch_results['test_acc'][-1])

        for key in ["train_loss","train_err", "train_acc"]: # average over the epoch
            epoch_val = np.mean(all_epoch_results[key][-len(train_dataloader):])
            all_epoch_results[f'{key}_epoch'].append(epoch_val)

        print(' Epoch %i. Test Loss%.3f, Err:%.3f'% (epoch_i, all_epoch_results['test_loss'][-1],
                                                     all_epoch_results['test_err'][-1]))

    if save_model == True:
        all_epoch_results['model_state'] = [model.state_dict()]

    return all_epoch_results


# -------------------------------------------
#   Build model based on flags
# -------------------------------------------
def build_model(flags, train_loader):
    """
    Function to build the different models we are using in experiments
    """
    if flags.layer_type =='MLP': LayerClass = DenseLayer
    elif flags.layer_type == 'LayerNorm': LayerClass = LayerNormLayer
    elif flags.layer_type == 'BatchNorm': LayerClass = BatchNormLayer
    elif flags.layer_type.startswith('EiShunt'): LayerClass = EiDenseWithShunt # DANN model
    elif flags.layer_type == 'ColumnEi':  LayerClass = ColummEiDense
    else: print('Layer type not recognised!'); raise

    #  Construct the layer_dims list:
    # ---------------------------------------------------
    input_dim = train_loader.dataset.n_pixels
    output_dim = train_loader.dataset.n_classes

    if flags.layer_type in ['MLP','LayerNorm','BatchNorm']:
        if flags.n_i > 0:
            print("WARNING: LayerClass is not EiShunt, but n_i is not 0. Setting n_i to zero")
            flags.n_i = 0
        hidden_dims = flags.n_e
    else:
        # set hidden dims and output_dim for the ei models
        hidden_dims = (flags.n_e, flags.n_i)
        if flags.layer_type == 'ColumnEi':
             output_dim = (output_dim, flags.n_i) # for colei hidden layers n_i is used as ratio to split units into e and i populations
        elif flags.layer_type.startswith('EiShunt'):
            output_dim = (output_dim, 1)  # always one output inhib (10%), even if multiple hidden inhib

    layer_dims = [input_dim] + [hidden_dims]*flags.n_hidden+ [output_dim]


    #  Build model:
    # ---------------------------------------------------
    if flags.NGD is False:
        model = build_densenet(DenseNet, LayerClass, layer_dims)
    else:  # Diagonal-NGD model
        class FisherLayerClass(FisherLayerMixin, LayerClass):
            pass

        model = build_densenet(FisherDenseNet, FisherLayerClass,layer_dims)
        if flags.layer_type == 'EiShunt' and flags.corrected_sgd:
            print('ERROR: cSGD and NGD should not be used together!')
            raise

        # the d-NGD model needs a few extra things
        model.scale_finv_by_norm_wex_finv = flags.scale_finv_by_norm_wex_finv # bool
        model.fisher_movingavg_scaling    = flags.f_ma_scale

        # assign a FisherCalculator() obj to each layer, and assign a f_lambda attr
        set_attr_for_all_layers(model, attr_name='fisher_calculator_object', class_def=FisherCalculator)
        for k, l in model.layers.items():
            l.f_lambda = flags.f_lambda

    # Apply some of the Eishunt configurations to model if applicable
    # ---------------------------------------------------
    if flags.layer_type == 'EiShuntBasic':
        for i, (key, layer) in enumerate(model.layers.items()):
            if flags.c_sgd:
                layer.update_policy =  DalesANN_cSGD_UpdatePolicy() # (cSGD_Mixin, DalesANN_SGD_UpdatePolicy)
            else:
                layer.update_policy =  DalesANN_SGD_UpdatePolicy() # (DalesANN_SGD_UpdatePolicy)

            layer.weight_init_policy = EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=flags.i_iid_i)

    elif flags.layer_type == 'EiShunt':
        for i, (key, layer) in enumerate(model.layers.items()):
            print(i, key, layer)
            # DalesANN_UpdatePolicy contains the following mixins
            # (cSGD_Mixin, ClipGradNorm_Mixin, ScaleLR_Mixin, DalesANN_SGD_UpdatePolicy)
            layer.update_policy = DalesANN_UpdatePolicy(lr_max = flags.lr,
                                                lr_min = flags.lr_min,
                                                lr_n = flags.lr_n,
                                                cosine_angle_epsilon = flags.cos_ep,
                                                max_grad_norm = flags.max_gn,
                                               )
            layer.weight_init_policy = EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=flags.i_iid_i)

    elif flags.layer_type =='ColumnEi':
        # set the first layer to be purely positive (the rest are ColumnEiWeightInitPolicy)
        model[0].weight_init_policy = PostiveWeightInitPolicy(dataset=flags.dataset)
        # we pass the dataset into Colei input layer as we use the pixel means to center acts
        # by default the param update policies are all ColumnEiSGD

    return model


# -------------------------------------------
#          Set up Experiment
# -------------------------------------------

# hardcode whether to overwrite experiment runs
if not in_notebook():
    overwrite_runs = True
else:
    overwrite_runs = True

flags = utils.FLAGS()

if not in_notebook():
    flags.update_from_command_line_args(verbose=False)

average_test_error = []
for seed in flags.seeds:
    flags.global_seed = seed
    if flags.run_exists('learning_curves.npy','exp_config.yaml') and not overwrite_runs:
        print("Exiting run as not overwriting \n")
        sys.exit()
    else:
         flags.save_path.mkdir(parents=True, exist_ok=True)

    print('\n ----- Running experiment ----- \n')
    flags.write_config_file_to_save_path('exp_config.yaml')

    device = utils.get_device()
    print('Running on device', device)
    print(flags)
    utils.set_seed_all(flags.global_seed)

    try:
        train_loader, eval_loader  = mnist.get_data(flags.dataset, flags.batch_size, seed=flags.global_seed,
                                                    flatten=True, to_device=True, test_set=flags.TEST)
    except:
        print('Failed to copy data to slurm_tmpdir... trying to load data locally')
        train_loader, eval_loader  = mnist.get_data(flags.dataset, flags.batch_size, seed=flags.global_seed,
                                                    flatten=True, to_device=True, test_set=flags.TEST,
                                                    directory='./data',copy_to_slurmtmpdir=False)

    eval_x, eval_y = utils.get_dataloader_xy(eval_loader)

    model = build_model(flags, train_loader)
    model.init_weights()
    model.to(device)
    print(model)

    # -------------------------------------------
    # Run training loop
    # -------------------------------------------
    results = main_loop(model, train_loader, eval_x, eval_y, flags, device, save_model=False)

    lc_path = str(flags.save_path / 'learning_curves.npy')
    print('\n ***********************')
    print(f'saving results to {lc_path}')
    print('***********************')
    np.save(lc_path, results)

    average_test_error.append(np.mean(results['test_err'][-5:]))

    # logging for orion
    results = []
    results += [dict(name='test_error',type='objective',value=np.mean(average_test_error))]
    orion_report_results(results)
