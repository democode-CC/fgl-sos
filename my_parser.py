import argparse


def parameter_parser():
    """
    A method to parse up command line parameters. By default it trains on the PubMed dataset.
    The default hyperparameters give a good quality representation without grid search.
    """
    parser = argparse.ArgumentParser(description="Run .")

    parser.add_argument('--edge_path', type=str,
                        default="/home/xxx/Sy/data/PemsD7/E_228.csv",
                        help="Edge list csv. PemsD7:/home/xxx/Sy/data/PemsD7/E_228.csv METR-LA:/home/xxx/Sy/METR-LA/STGCN/data/adj_mat.npy")

    parser.add_argument("--dataset",
                        type=str,
                        default="cora",
                        help="cora, citeseer, pubmed, polblogs")
    
    parser.add_argument("--cluster-number",
                        type=int,
                        default=4,
                        help="Number of clusters extracted. Default is 10. In FL, it also represent the number of clients")
    
    parser.add_argument("--label-number",
                        type=int,
                        default=2,
                        help="The dimension of the node embedding. Default is 2.")


    parser.add_argument('--only-publish',
                        type=bool,
                        default=False,
                        help='If True, only publish the subgraph, do not train.')
    
    parser.add_argument('--MI_method',
                        type=str,
                        default='MINE',
                        help='MINE, DIM')

    parser.add_argument("--framework",
                        type=str,
                        default='FedAvg',
                        help="FedAvg, //FedQSGD (work), FedSpar, FedSIGN, FedTopK (work),// FedAvg, FedAvg, Centralized FedP")

    # parser.add_argument("--partial",
    #                     type=str,
    #                     default=True,
    #                     help="Partial FedAvg")


    parser.add_argument("--model",
                        type=str,
                        default='GCN',
                        help="The adopted model: STGCN, STGAT, gwnet, DCRNN, MTGNN, TGCN.")

    parser.add_argument("--epoch",
                        type=int,
                        default=50,
                        help="Number of training epochs. Default is 50.")

    parser.add_argument("--batch",
                        type=int,
                        default=50,
                        help="Number of training batch. Default is 50.")

    parser.add_argument("--seed",
                        type=int,
                        default=99,
                        help="Random seed for train-test split. Default is 42 103 72 100 85.")

    parser.add_argument("--dropout",
                        type=float,
                        default=0.1,
                        help="Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.01,
                        help="Learning rate. Default is 0.0001 (STGCN) 0.0002 (STGAT) 2e-6 (graph wavenet).")

    parser.add_argument("--weight_decay",
                        type=float,
                        default=5e-4,
                        help="weight_decay 5e-4")

    parser.add_argument("--device",
                        type=str,
                        default='cuda:0',
                        help="The GPU is used.")



    # -----------------------------

    parser.add_argument("--clustering-method",
                        nargs="?",
                        default="metis",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")


    # parser.add_argument("--nodes_number",
    #                     type=int,
    #                     default=228,
    #                     help="The total number of nodes = 1159 (Beijing) 1335 (Shanghai) 228 (Pems) 207 (METR-LA)")

    

    parser.add_argument('--hidden',
                        type=int,
                        default=12,
                        help='Number of hidden units.')

    parser.add_argument('--n_heads',
                        type=int,
                        default=8,
                        help='Number of head attentions.')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.2,
                        help='Alpha for the leaky_relu.')

    parser.add_argument("--block",
                        type=str,
                        default='1_32_64_64_32_128')

    # -------------Federated Learning parameters----------------

    parser.add_argument('--fed',
                        type=float,
                        default=True,
                        help='If True, execute federated learning.')

    # parser.add_argument('--num_users',
    #                     type=int,
    #                     default=4,
    #                     help="number of users: K")

    parser.add_argument('--frac',
                        type=float,
                        default=0.4,
                        help="the fraction of clients: C")

    parser.add_argument('--local_epoch',
                        type=int,
                        default=1,
                        help="the number of local epochs: E")

    parser.add_argument('--local_batch',
                        type=int,
                        default=50,
                        help="local batch size: B")

    parser.add_argument('--noise_type',
                        type=str,
                        default='laplace',
                        help="gaussian or laplace")

    parser.add_argument('--noise_ratio',
                        type=float,
                        default=0.4,
                        help="local batch size: B")

    parser.add_argument('--error_rate',
                        type=float,
                        default=0.4,
                        help='packet loss rate.')


    # -------------IB parameters----------------
    parser.add_argument('--IB',
                        type=float,
                        default=True,
                        help='If True, execute IB. If False, execute Ori')

    # parser.add_argument('--IB_method',
    #                     type=float,
    #                     default='SIB',
    #                     help='GIB, SIB.')


    # -------------Attack parameters----------------
    parser.add_argument('--attack',
                        type=float,
                        default=True,
                        help='If True, execute attack.')

    parser.add_argument('--attack_method',
                        type=str,
                        default='dice',
                        help='random, mettack, dice')

    parser.add_argument("--perturbation",
                        type=int,
                        default=800,
                        help="Number of flipped edges. Default is 50.")

    parser.add_argument("--scale",
                        type=int,
                        default=20,
                        help="scale up of attacker 's upload.")





    return parser.parse_args()



