import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import os
from model.GCN import GCN
from model.GCN_IB import GCN_IB, Global_Discriminator, Local_Discriminator
from Fed import FedAvg
import yaml
import copy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from Adam_Half import Adam16
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import random
from utils.math_utils import MAPE, MAE, RMSE

class Trainer(object):
    """
    Trainer.
    """

    def __init__(self, args, clustrMachine, matrix):
        """
        Define the trainer
        :param args: The args from parser
        :param edges: The Edges List, size = [2, number_of_edges]
        :param matrix: The Adjacent Matrix(treated). size = [number_of_nodes, number_of_nodes]
        :param data: The DataSet defined in utils
        """

        self.args = args
        self.number_nodes = self.args.nodes_number
        self.clustrMachine = clustrMachine
        self.model = {}
        self.optimizer = {}
        self.matrix = {}
        self.matrix_gwnet = matrix
        self.built_block()
        self.create_model()
        self.val_mape = []
        self.best_mape = [float("+inf") for i in range(args.cluster_number)]
        self.path = 'save/' + str(args.cluster_number) + '/' + str(self.args.n_heads) + '/'
        self.res_path = 'result/' + str(self.args.cluster_number) + '/' + str(args.n_heads) + '.csv'
        self.gen_path()
        # for i in range(0, 8):
        #     print(len(clustrMachine.sg_nodes[i]))

    def gen_path(self):
        os.makedirs('result/', exist_ok=True)
        os.makedirs('result/' + str(self.args.cluster_number) + '/', exist_ok=True)
        os.makedirs('save/', exist_ok=True)
        os.makedirs('save/' + str(self.args.cluster_number) + '/', exist_ok=True)
        os.makedirs('save/' + str(self.args.cluster_number) + '/' + str(self.args.n_heads) + '/', exist_ok=True)
        f = open(self.res_path, "w+")
        f.write('RMSE,MAE,MAPE,TIME\n')
        f.close()

    def built_block(self):
        block = [[], []]
        b = self.args.block.split('_')
        block[0].append(int(b[0]))
        block[0].append(int(b[1]))
        block[0].append(int(b[2]))
        block[1].append(int(b[3]))
        block[1].append(int(b[4]))
        block[1].append(int(b[5]))
        self.blocks = block

    def create_model(self):
        """
        Creating a model to CPU/GPU.
        """
        n_his = self.args.bash
        hidden = self.args.hidden
        drop_prob = self.args.dropout
        n_heads = self.args.n_heads
        alpha = self.args.alpha
        device = self.args.device
        n_pred = self.args.pred


        for cluster in self.clustrMachine.clusters:
            self.matrix[cluster] = self.clustrMachine.sg_matrix[cluster]
            self.model[cluster].type(torch.cuda.HalfTensor)
            self.model[cluster] = self.model[cluster].to(self.args.device)
            self.optimizer[cluster] = Adam16(self.model[cluster].parameters(), lr=self.args.learning_rate,
                                                       weight_decay=self.args.weight_decay)


    def train_epoch(self, batch_size, cluster):
        """
        The Process Of Training a epoch
        :param batch_size: The Batch number
        :return: The Average MSE Lose
        """
        epoch_training_losses = []
        self.model[cluster].train()


        data = self.clustrMachine.sg_data[cluster]
        pred_list = [i for i in range(data['train_x'].shape[0])]
        random.shuffle(pred_list)
        for i in range(0, data['train_x'].shape[0], batch_size):
            idc = pred_list[i: i + batch_size]
            self.optimizer[cluster].zero_grad()
            input = torch.unsqueeze(data['train_x'][idc], 3)

            if self.args.model == 'gwnet':
                output = data['train_y'][idc].unsqueeze(1)
            else:
                output = data['train_y'][idc]

            out = self.model[cluster](input)
            loss = self.loss_criterion(out, output)
            loss.backward()
            self.optimizer[cluster].step()
            epoch_training_losses.append(loss.detach().cpu().numpy())
        return sum(epoch_training_losses)


    def train(self):
        """
        Training a model.
        """
        self.loss_criterion = nn.MSELoss()
        train_loss = []
        val_loss = []
        for i in range(0, self.args.epoch):
            opoch_start = timer()
            loss_t = 0
            for cluster in self.clustrMachine.clusters:
                loss_t += self.train_epoch(self.args.batch, cluster)
            opoch_end = timer()
            train_loss.append(loss_t / self.number_nodes)

            opoch_time = opoch_end - opoch_start
            print('This is the {}round. The training time of the epoch {:.2f}s'.format(i, opoch_time))
            loss_v = self.val(opoch_time)
            val_loss.append(loss_v)

        plt.plot(train_loss, label="training loss")
        plt.plot(val_loss, label="validation loss")
        plt.legend()
        plt.show()

    def val(self, opoch_time):
        """
        Test on val when training
        """
        mse = 0
        mape = 0
        mae = 0
        rmse = 0
        for cluster in self.clustrMachine.clusters:
            num = self.clustrMachine.sg_matrix[cluster].shape[1]
            std = self.clustrMachine.sg_data[cluster].std
            mean = self.clustrMachine.sg_data[cluster].mean
            pred_list = torch.rand(self.clustrMachine.sg_data[cluster]['val_y'].shape).to(self.args.device)
            with torch.no_grad():
                self.model[cluster].eval()
                for i in range(0, len(self.clustrMachine.sg_data[cluster]['val_x']), self.args.batch):
                    input = self.clustrMachine.sg_data[cluster]['val_x'][i: i + self.args.batch, :, :]
                    input = torch.unsqueeze(input, 3)
                    pred_list[i: i + self.args.batch, :, :] = self.model[cluster](input).squeeze()

                out_list = self.clustrMachine.sg_data[cluster]['val_y']
                mse += self.loss_criterion(pred_list, out_list) * num
                y_true = out_list.detach().cpu().numpy() * std + mean
                y_pred = pred_list.detach().cpu().numpy() * std + mean
                bmape = MAPE(y_true, y_pred) * num
                bmae = MAE(y_true, y_pred) * num
                brmse = RMSE(y_true, y_pred) * num
                mape += bmape
                mae += bmae
                rmse += brmse
                if self.best_mape[cluster] > bmape:
                    self.best_mape[cluster] = bmape
                    torch.save(self.model[cluster].state_dict(), 'save/test1/learned_model_{}_{}_{}.pkl'.format(self.args.dataset, self.args.framework, self.args.model))
        mse /= self.number_nodes
        mape /= self.number_nodes
        mae /= self.number_nodes
        rmse /= self.number_nodes
        self.val_mape.append(mape)
        print('RMSE:{:.2f}; MAE:{:.2f}; MAPE:{:.2f}'.format(rmse, mae, mape))
        with open(self.res_path, 'a') as to_file:
            to_file.write('{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(rmse, mae, mape, opoch_time))
            to_file.close()
        return mse

    def test(self):
        """
        Testing
        """

        mape = 0
        mae = 0
        rmse = 0
        for cluster in self.clustrMachine.clusters:
            self.model[cluster].load_state_dict(torch.load('save/test1/learned_model_{}_{}_{}.pkl'.format(self.args.dataset, self.args.framework, self.args.model)))
            num = self.clustrMachine.sg_matrix[cluster].shape[1]
            std = self.clustrMachine.sg_data[cluster].std
            mean = self.clustrMachine.sg_data[cluster].mean
            pred_list = torch.rand(self.clustrMachine.sg_data[cluster]['test_y'].shape).to(self.args.device)
            with torch.no_grad():
                self.model[cluster].eval()
                for i in range(0, len(self.clustrMachine.sg_data[cluster]['test_x']), self.args.batch):
                    input = self.clustrMachine.sg_data[cluster]['test_x'][i: i + self.args.batch, :, :]
                    input = torch.unsqueeze(input, 3)
                    pred_list[i: i + self.args.batch, :, :] = self.model[cluster](input).squeeze()

                out_list = self.clustrMachine.sg_data[cluster]['test_y']
                y_true = out_list.detach().cpu().numpy() * std + mean
                y_pred = pred_list.detach().cpu().numpy() * std + mean
                mape += MAPE(y_true, y_pred) * num
                mae += MAE(y_true, y_pred) * num
                rmse += RMSE(y_true, y_pred) * num
        mape /= self.number_nodes
        mae /= self.number_nodes
        rmse /= self.number_nodes
        print('*' * 50)
        print('Test DataSet:')
        print('RMSE:{:.2f}; MAE:{:.2f}; MAPE:{:.2f}'.format(rmse, mae, mape))
        count = np.array([_ for _ in range(0, self.args.epoch)])

        np.save('result/test1/learning_curve_{}_{}_{}.npy'.format(self.args.dataset, self.args.model, self.args.framework), self.val_mape)
        print(self.val_mape)
