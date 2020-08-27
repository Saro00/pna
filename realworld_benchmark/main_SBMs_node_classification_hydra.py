"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm
import hydra


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from data.SBMs import SBMsDataset
from nets.SBMs_node_classification.eig_net import EIGNet
from train.train_SBMs_node_classification import train_epoch_sparse as train_epoch, \
    evaluate_network_sparse as evaluate_network

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id, verbose=True):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        if verbose:
            print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        if verbose:
            print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(net_params, verbose=True):
    model = EIGNet(net_params)
    total_param = 0
    if verbose:
        print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    if verbose:
        print('MODEL/Total parameters:', total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    start0 = time.time()
    per_epoch_time = []

    DATASET_NAME = dataset.name

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    if MODEL_NAME in ['GatedGCN']:
        if net_params['pos_enc']:
            print("[!] Adding graph positional encoding.")
            dataset._add_positional_encodings(net_params['pos_enc_dim'])
            print('Time PE:', time.time() - start0)

    trainset, valset, testset = dataset.train, dataset.val, dataset.test

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    if hydra.is_first_execution():
        print("Training Graphs: ", len(trainset))
        print("Validation Graphs: ", len(valset))
        print("Test Graphs: ", len(testset))
        print("Number of Classes: ", net_params['n_classes'])

    model = EIGNet(net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)

    if hydra.is_first_execution():
        start_epoch = 0
    else:
        start0 -= hydra.retrieved_checkpoint.time_elapsed
        start_epoch = hydra.retrieved_checkpoint.last_epoch
        states = torch.load(hydra.retrieved_checkpoint.linked_files()[0])
        model.load_state_dict(states['model'])
        optimizer.load_state_dict(states['optimizer'])
        scheduler.load_state_dict(states['scheduler'])

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []

    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    last_hydra_checkpoint = start0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(start_epoch, params['epochs']), mininterval=params['hydra_progress_bar_every'],
                  maxinterval=None, unit='epoch', initial=start_epoch, total=params['epochs']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                if MODEL_NAME in ['RingGNN', '3WLGNN']:  # since different batch training function for dense GNNs
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                               epoch, params['batch_size'])
                else:  # for all other models common train function
                    epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                               epoch)

                epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader, epoch)
                _, epoch_test_acc = evaluate_network(model, device, test_loader, epoch)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)
                print('')

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR SMALLER OR EQUAL TO MIN LR THRESHOLD.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - start0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break

                # Saving checkpoint
                if hydra.is_available() and (time.time() - last_hydra_checkpoint) > params[
                    'hydra_checkpoint_every']:
                    last_hydra_checkpoint = time.time()
                    ck_path = '/tmp/epoch_{}.pkl'.format(epoch + 1)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, ck_path)
                    ck = hydra.checkpoint()
                    ck.last_epoch = epoch + 1
                    ck.time_elapsed = time.time() - start0
                    # save best epoch
                    ck.link_file(ck_path)
                    ck.save_to_server()

                    if hydra.is_available() and epoch % params['hydra_eta_every'] == 0:
                        hydra.set_eta(per_epoch_time[-1] * (params['epochs'] - epoch - 1))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    _, test_acc = evaluate_network(model, device, test_loader, epoch)
    _, val_acc = evaluate_network(model, device, val_loader, epoch)
    _, train_acc = evaluate_network(model, device, train_loader, epoch)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("Train Accuracy: {:.4f}".format(train_acc))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    if hydra.is_available():
        hydra.save_output({'loss': {'train': epoch_train_losses, 'val': epoch_val_losses},
                           'acc': {'train': epoch_train_acc, 'val': epoch_val_acc}}, 'history')
        hydra.save_output(
            {'test_acc': test_acc, 'train_acc': train_acc, 'val_acc': val_acc, 'total_time': time.time() - start0,
             'avg_epoch_time': np.mean(per_epoch_time)}, 'summary')


    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY: {:.4f}\nTRAIN ACCURACY: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        test_acc, train_acc, epoch, (time.time() - start0) / 3600, np.mean(per_epoch_time)))


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--hidden_dim', help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--JK', defaul='last', help='Jumping Knowledge')
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--pos_enc', help="Please give a value for pos_enc")

    # hydra params
    parser.add_argument('--hydra', action='store_true', default=False, help='Run in Hydra environment.')
    parser.add_argument('--hydra_checkpoint_every', type=int, default=100, help='Save checkpoints to hydra every.')
    parser.add_argument('--hydra_eta_every', type=int, default=100, help='Update ETA to hydra every.')
    parser.add_argument('--hydra_progress_bar_every', type=float, default=1,
                        help='Update progress hydra every (seconds).')

    # eig params
    parser.add_argument('--aggregators', type=str, help='Aggregators to use.')
    parser.add_argument('--scalers', type=str, help='Scalers to use.')
    parser.add_argument('--NN_eig', action='store_true', default=False, help='NN eig aggr.')
    parser.add_argument('--towers', type=int, default=5, help='Towers to use.')
    parser.add_argument('--divide_input_first', type=bool, help='Whether to divide the input in first layer.')
    parser.add_argument('--divide_input_last', type=bool, help='Whether to divide the input in last layers.')
    parser.add_argument('--gru', type=bool, help='Whether to use gru.')
    parser.add_argument('--edge_dim', type=int, help='Size of edge embeddings.')
    parser.add_argument('--pretrans_layers', type=int, help='pretrans_layers.')
    parser.add_argument('--posttrans_layers', type=int, help='posttrans_layers.')
    parser.add_argument('--not_pre', action='store_true', default=False, help='Not applying pre-transformation')

    args = parser.parse_args()

    # hydra load
    if args.hydra:
        print('I am passing here 1')
        if not hydra.is_available():
            print('hydra: not available')
            args.hydra = False

    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'], verbose=hydra.is_first_execution())
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = SBMsDataset(DATASET_NAME, verbose=hydra.is_first_execution())
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)

    #hydra parameters
    params['hydra'] = args.hydra
    params['hydra_checkpoint_every'] = args.hydra_checkpoint_every
    params['hydra_eta_every'] = args.hydra_eta_every
    params['hydra_progress_bar_every'] = args.hydra_progress_bar_every

    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.hidden_dim is not None:
        net_params['hidden_dim'] = int(args.hidden_dim)
    if args.out_dim is not None:
        net_params['out_dim'] = int(args.out_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.JK is not None:
        net_params['JK'] = args.JK
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.pos_enc is not None:
        net_params['pos_enc'] = True if args.pos_enc == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)
    if args.aggregators is not None:
        net_params['aggregators'] = args.aggregators
    if args.scalers is not None:
        net_params['scalers'] = args.scalers
    if args.towers is not None:
        net_params['towers'] = args.towers
    if args.divide_input_first is not None:
        net_params['divide_input_first'] = args.divide_input_first
    if args.divide_input_last is not None:
        net_params['divide_input_last'] = args.divide_input_last
    if args.NN_eig is not None:
        net_params['NN_eig'] = args.NN_eig
    if args.gru is not None:
        net_params['gru'] = args.gru
    if args.edge_dim is not None:
        net_params['edge_dim'] = args.edge_dim
    if args.pretrans_layers is not None:
        net_params['pretrans_layers'] = args.pretrans_layers
    if args.posttrans_layers is not None:
        net_params['posttrans_layers'] = args.posttrans_layers
    if args.not_pre is not None:
        net_params['not_pre'] = args.not_pre

    # SBM

    net_params['in_dim'] = torch.unique(dataset.train[0][0].ndata['feat'], dim=0).size(
        0)  # node_dim (feat is an integer)
    net_params['n_classes'] = torch.unique(dataset.train[0][1], dim=0).size(0)

    # D = torch.sparse.sum(dataset.graph.adjacency_matrix(transpose=True), dim=-1).to_dense()
    # net_params['avg_d'] = dict(lin=torch.mean(D),
    # exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
    # log=torch.mean(torch.log(D + 1)))

    num_nodes = [dataset.train[i][0].number_of_nodes() for i in range(len(dataset.train))]
    net_params['avg_d'] = int(np.ceil(np.mean(num_nodes)))  # wrong!!!!

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    net_params['total_param'] = view_model_param(net_params, verbose=hydra.is_first_execution())
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()
