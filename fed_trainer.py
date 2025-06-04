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
from prettytable import PrettyTable
import yaml
import copy
from Adam_Half import Adam16
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torch_geometric.nn.conv.gcn_conv import gcn_norm

# Attack
from attack import random_attack, meta_attack, dice_attack


# Debug
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])



class Fed_Trainer(object):
    """
    Trainer.
    """
    def __init__(self, args, clustrMachine, data):
        """
        Define the trainer
        :param args: The args from parser
        :param edges: The Edges List, size = [2, number_of_edges]
        :param matrix: The Adjacent Matrix(treated). size = [number_of_nodes, number_of_nodes]
        :param data: The DataSet defined in utils
        """

        self.args = args
        # self.number_nodes = self.args.nodes_number
        self.clustrMachine = clustrMachine
        self.data = data.to(self.args.device)
        self.data.x = self.data.x.half()

        self.in_num = clustrMachine.features.shape[1]
        self.out_num = len(set(clustrMachine.labels))
        self.number_clients = self.args.cluster_number
        self.model = {}
        self.discriminator = {}
        self.client_model = {}
        self.IB_model = {}
        for i in range(args.cluster_number):
            self.client_model[i] = {}
        self.client_model_feature = {}
        for i in range(args.cluster_number):
            self.client_model_feature[i] = {}
        self.optimizer = {}
        self.matrix = {}
        self.create_model()
        self.local_data = {}
        # self.val_mape = []
        # self.best_mape = [float("+inf") for i in range(args.cluster_number)]
        self.path = 'save/' + str(args.cluster_number) + '/' + str(self.args.n_heads) + '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.res_path = 'result/' + str(self.args.cluster_number) + '/' + str(args.cluster_number) + '.csv'
        self.gen_path()
        # for i in range(0, 8):
        #     print(len(clustrMachine.sg_nodes[i]))

        # Val
        self.val_client_acc_list = []
        self.val_client_loss_list = []

        self.val_global_loss_list = []

        for cluster in self.clustrMachine.clusters:
            self.val_client_acc_list.append([])

        # Attack
        self.attacker_id = 0




    def gen_path(self):
        os.makedirs('result/', exist_ok=True)
        os.makedirs('result/' + str(self.args.cluster_number) + '/', exist_ok=True)
        os.makedirs('save/', exist_ok=True)
        os.makedirs('save/' + str(self.args.cluster_number) + '/', exist_ok=True)
        os.makedirs('save/' + str(self.args.cluster_number) + '/' + str(self.args.cluster_number) + '/', exist_ok=True)
        f = open(self.res_path, "w+")
        f.write('RMSE,MAE,MAPE,TIME\n')
        f.close()


    def create_model(self): # Modified here because each client has different number of nodes
        """
        Creating a model to CPU/GPU.
        """
        # Create Model
        if self.args.model == 'GCN':
            self.global_model = GCN(self.in_num, self.out_num).to(self.args.device).half()
            if self.args.IB == True:
                for cluster in self.clustrMachine.clusters:
                    self.IB_model[cluster] = GCN_IB(self.in_num, self.out_num).to(self.args.device).half()



    def publish_subgraph(self, cluster):
        self.global_d = Global_Discriminator(self.args).to(self.args.device).half()
        self.local_d = Local_Discriminator(self.args).to(self.args.device).half()
        optimizer = Adam16(self.IB_model[cluster].parameters(), lr=0.001,
                           weight_decay=5*10**-5)
        optimizer_local = Adam16(self.local_d.parameters(),
                                 lr=0.001,
                                 weight_decay=5*10**-5)
        optimizer_global = Adam16(self.global_d.parameters(),
                                  lr=0.001,
                                  weight_decay=5*10**-5)

        for i in range(10):
            _, node_embedding, graph_embedding, positive, negative,  pos_penalty, out = self.IB_model[cluster](self.local_data[cluster])
            for j in range(50):
                # local
                optimizer_local.zero_grad()
                optimizer_global.zero_grad()

                DIM_loss = - self.MI_Est(node_embedding, graph_embedding, positive, beta=0.5, gamma=0.5, est=self.args.MI_method)
                DIM_loss.backward(retain_graph=True)
                optimizer_local.step()
                optimizer_global.step()

            mi_loss = self.MI_Est(node_embedding, graph_embedding, positive, beta=0.5, gamma=0.5, est=self.args.MI_method)
            cls_loss = F.nll_loss(out[self.local_data[cluster].train_mask], self.local_data[cluster].y[self.local_data[cluster].train_mask])
            optimizer.zero_grad()
            loss = cls_loss + pos_penalty + 0.2 * mi_loss
            loss.backward()
            optimizer.step()

        # Subgraph Publish
        new_edge_index, _, _, _, _, _, _ = self.IB_model[cluster](self.local_data[cluster])
        return new_edge_index

            # print("Loss:%.2f"%(loss))





    def count_parameters(self, model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def train(self):
        """
        Training a model.
        """
        train_loss = []
        val_loss_list = []
        val_acc_list = []
        self.Mi_loss = 0

        # w_glob_list = []


        for i in range(0, self.args.epoch):
            self.global_model.train()
            opoch_start = timer()
            num_params = 0
            for param in self.global_model.parameters():
                num_params += param.numel()
            w_glob = self.train_client(i)
            self.global_model.load_state_dict(w_glob)
            opoch_end = timer()
            # train_loss.append(loss_t / self.number_nodes)
            opoch_time = opoch_end - opoch_start
            print('This is the {}round. The training time of the epoch {:.2f}s'.format(i, opoch_time))
            val_loss, val_acc = self.val()
            print('acc', val_acc)
            print('loss', val_loss)

            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

        if self.args.IB == True:
            if self.args.attack == True:
                np.save('att_ib_acc.npy', np.array(val_acc_list))
                np.save('att_ib_loss.npy', np.array(val_loss_list))
            else:
                np.save('ib_acc.npy', np.array(val_acc_list))
                np.save('ib_loss.npy', np.array(val_loss_list))
                np.save('client_ib_acc.npy', np.array(self.val_client_acc_list[0]))
        else:
            if self.args.attack == True:
                np.save('att_ori_acc.npy', np.array(val_acc_list))
                np.save('att_ori_loss.npy', np.array(val_loss_list))
            else:
                np.save('ori_acc.npy', np.array(val_acc_list))
                np.save('ori_loss.npy', np.array(val_loss_list))
                np.save('client_ori_acc.npy', np.array(self.val_client_acc_list[0]))
        plt.plot(val_acc_list, label="validation loss")
        plt.legend()
        plt.savefig('loss.png')
        plt.show()

    def train_client(self, global_i):
        """
        The Process Of Training a epoch
        :param batch_size: The Batch number
        :return: w_glob
        """
        w_locals = []
        embedding_locals = []
        positive_locals = []
        for cluster in self.clustrMachine.clusters:
            self.val_client_acc_list.append([])

            # Load local data
            if global_i == 0:
                if cluster == self.attacker_id:
                    self.local_data[cluster] = copy.deepcopy(self.clustrMachine.sg_data[cluster])
                else:
                    self.local_data[cluster] = copy.deepcopy(self.clustrMachine.sg_data[cluster])
                    # Publish subgraph
                    self.local_data[cluster].edge_index = self.publish_subgraph(cluster)
                    
                    # Save the published subgraph for analysis
                    published_data = copy.deepcopy(self.local_data[cluster])
                    self.clustrMachine.sg_data[cluster] = published_data
                    self.clustrMachine.save_clusters(save_dir='saved_ib_clusters')
                    
                    if self.args.only_publish == True:
                        continue



            # local training
            self.model[cluster] = copy.deepcopy(self.global_model)
            self.model[cluster].half()
            self.model[cluster].train()
            optimizer = Adam16(self.model[cluster].parameters(), lr=self.args.learning_rate,
                               weight_decay=self.args.weight_decay)



            # Poisoning Attack
            if global_i == 0:
                if self.args.attack == True:
                    if cluster == self.attacker_id:
                        if self.args.attack_method == 'random':
                            modified_adj = random_attack(self.local_data[cluster], perturbation_num=self.args.perturbation)
                            self.local_data[cluster].edge_index = pyg.utils.dense_to_sparse(modified_adj.to(self.args.device))[0]
                        elif self.args.attack_method == 'dice':
                            modified_adj = dice_attack(self.local_data[cluster], perturbation_num=self.args.perturbation)
                            self.local_data[cluster].edge_index = pyg.utils.dense_to_sparse(modified_adj.to(self.args.device))[0]
                        elif self.args.attack_method == 'mettack':
                            modified_adj = meta_attack(self.args, self.model[cluster], self.local_data[cluster], perturbation_num=self.args.perturbation)
                            self.local_data[cluster].edge_index = pyg.utils.dense_to_sparse(modified_adj.to(self.args.device))[0]


            # Begin train
            # client epoch
            for iter in range(self.args.local_epoch):
                # self.local_data[cluster] = self.clustrMachine.sg_data[cluster]
                # data.adj_t = gcn_norm(data.adj_t) # adj normalization
                optimizer.zero_grad()
                out = self.model[cluster](self.local_data[cluster])
                loss = F.nll_loss(out[self.local_data[cluster].train_mask], self.local_data[cluster].y[self.local_data[cluster].train_mask])
                loss.backward()
                optimizer.step()

            # val
            val_client_loss, val_client_acc = self.val_client(cluster)
            self.val_client_acc_list[cluster].append(val_client_acc)
            print(val_client_acc)



            if self.args.attack == True:
                if cluster == self.attacker_id:
                    for param in self.model[cluster].state_dict():
                        self.model[cluster].state_dict()[param] = self.args.scale * self.model[cluster].state_dict()[param]
                    w = self.model[cluster].state_dict()
                else:
                    w = self.model[cluster].state_dict()
            else:
                w = self.model[cluster].state_dict()
            w_locals.append(copy.deepcopy(w))

            # embedding_locals.append(graph_embedding)
            # positive_locals.append(positive)

        # Central server do:
        # aggregate and update global weights
        w_glob = FedAvg(w_locals)

        # # Donsker-Varadhan IB
        # embeddings = torch.cat(tuple(embedding_locals), dim=0)
        # positive = torch.cat(tuple(positive_locals), dim=0)
        # self.optimizer_discriminator.zero_grad()
        # self.Mi_loss = - self.MI_Est(self.discriminator, embeddings, positive)
        # self.Mi_loss.backward(retain_graph = True)
        # print('mi', self.Mi_loss)
        # self.optimizer_discriminator.step()


        return w_glob

    # @torch.no_grad()
    def val(self):
        """
        Test on val when training
        """
        val_model = copy.deepcopy(self.global_model)
        val_model.eval()
        with torch.no_grad():
            val_out = val_model.forward(self.data)
            pred = val_out.argmax(dim=-1)
            val_loss = F.nll_loss(val_out[self.data.val_mask], self.data.y[self.data.val_mask])
            val_loss = val_loss.detach().cpu().numpy()
            val_acc = int((pred[self.data.val_mask] == self.data.y[self.data.val_mask]).sum()) / int(self.data.val_mask.sum())
            return val_loss, val_acc



    def val_client(self, cluster):
        """
        Test on val when training
        """
        val_model = copy.deepcopy(self.model[cluster])
        val_model.eval()
        with torch.no_grad():

            val_out = val_model.forward(self.data)
            pred = val_out.argmax(dim=-1)
            val_loss = F.nll_loss(val_out[self.data.val_mask], self.data.y[self.data.val_mask])
            val_loss = val_loss.detach().cpu().numpy()
            val_acc = int((pred[self.data.val_mask] == self.data.y[self.data.val_mask]).sum()) / int(self.data.val_mask.sum())
            self.val_global_loss_list.append(val_loss)
            return val_loss, val_acc


    def test(self):
        """
        Testing
        """
        test_model = copy.deepcopy(self.global_model)
        test_model.eval()
        with torch.no_grad():
            test_out = test_model.forward(self.data)
            pred = test_out.argmax(dim=-1)
            test_loss = F.nll_loss(test_out[self.data.test_mask], self.data.y[self.data.test_mask])
            test_loss = test_loss.detach().cpu().numpy()
            test_acc = int((pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()) / int(self.data.test_mask.sum())
            print('final ACC', test_acc)


    # for IB
    def MI_Est(self, discriminator, embeddings, positive):
        # embeddings torch.Size([1, 7])
        shuffle_embeddings = embeddings[torch.randperm(embeddings.shape[0])] # Returns a randomly shuffled array of indices from 0 to n-1
        # discriminator  f_phi_2
        # positive G_sub
        # embedding G
        joint = discriminator(embeddings,positive)
        margin = discriminator(shuffle_embeddings,positive)
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(margin)))
        return mi_est

    def MI_Est(self, node_embedding, graph_embedding, positive, beta, gamma, est=None):
        if est == 'MINE':
            return self.MINE_Est(node_embedding, graph_embedding, positive, beta, gamma)
        elif est == 'DIM':
            return self.DIM_Est(node_embedding, graph_embedding, positive, beta, gamma)
        else:
            raise ValueError(f"Invalid estimation method: {est}")


    def DIM_Est(self, node_embedding, graph_embedding, positive, beta, gamma):
        self.beta = beta
        self.gamma = gamma

        # Local
        sampled_node_embedding = node_embedding[torch.randperm(node_embedding.shape[0])]
        Ej = -F.softplus(-self.local_d(node_embedding, positive)).mean()
        Em = F.softplus(self.local_d(sampled_node_embedding, positive)).mean()
        LOCAL = (Em - Ej) * self.beta

        # Global
        Ej = -F.softplus(-self.global_d(graph_embedding, positive)).mean()
        Em = F.softplus(self.global_d(graph_embedding, positive)).mean()
        GLOBAL = (Em - Ej) * self.gamma

        return LOCAL + GLOBAL

    def MINE_Est(self, node_embedding, graph_embedding, positive, beta, gamma):
        """
        Mutual Information Neural Estimation (MINE) implementation
        Args:
            node_embedding: Node embeddings
            graph_embedding: Graph embeddings
            positive: Positive samples
            beta: Weight for local MI
            gamma: Weight for global MI
        """
        self.beta = beta
        self.gamma = gamma

        # Local MINE
        sampled_node_embedding = node_embedding[torch.randperm(node_embedding.shape[0])]
        joint_local = self.local_d(node_embedding, positive)
        marginal_local = self.local_d(sampled_node_embedding, positive)
        LOCAL = (joint_local.mean() - torch.log(torch.exp(marginal_local).mean())) * self.beta

        # Global MINE
        joint_global = self.global_d(graph_embedding, positive)
        marginal_global = self.global_d(graph_embedding, positive)
        GLOBAL = (joint_global.mean() - torch.log(torch.exp(marginal_global).mean())) * self.gamma

        return LOCAL + GLOBAL


