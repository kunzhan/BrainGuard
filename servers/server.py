import copy
import numpy as np
import torch
import time
from clients.client import *
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import os
from utils.utils import Clipper
from utils.nsd_access import NSDAccess


class Server(object):
    def __init__(self, args):
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = [1,2,5,7]
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(len(self.num_clients) * self.join_ratio)
        self.cuda_id = args.cuda_id
        self.clients = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_loss = []
        self.rs_train_loss = []

        self.writer = None
        self.set_clients(args, Client)
        if not args.resume:
            log_dir = './logs/{}/{}'.format(args.train_type, time.strftime("%b%d_%d-%H-%M", time.localtime()))
            self.writer = SummaryWriter(log_dir=log_dir)
            os.makedirs(log_dir, exist_ok=True)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {len(self.num_clients)}")
        print("Finished creating server and clients.")
        self.selected_clients = self.select_clients()
        self.prompts_list = self.prepare_coco(args.data_root)
        self.Budget = []

    def train(self, args):
        for i in range(self.global_rounds):
            args.logger.info(f"============= Round: {i+1}th =============")
            s_t = time.time()
            # self.selected_clients = self.select_clients()
            if i != 0 and i < 600:
                self.send_models(i)
                # if i % args.eval_interval == 0:
                #     for client in self.selected_clients:
                #         client.eval_local_model(self.writer, i, args.logger, self.prompts_list)

            thread_train = [Thread(target=client.train, args=(self.writer, i, args.logger))
                    for client in self.selected_clients]
            [t.start() for t in thread_train]
            [t.join() for t in thread_train]

            # for client in self.selected_clients:
            #     client.train(self.writer, i, args.logger)
            if i < 600:
                self.receive_models()
                self.aggregate_parameters()
                
                # eval global model after aggregate_parameters
                # threads_eval_global = [Thread(target=client.eval_global_model, args=(self.global_model, self.writer, i, args.logger))
                #         for client in self.selected_clients]
                # [t.start() for t in threads_eval_global]
                # [t.join() for t in threads_eval_global]
                if i % args.eval_interval == 0:
                    args.logger.info(f'======Start using clients data eval global model======')
                    for client in self.selected_clients:
                        client.eval_global_model(self.global_model, self.writer, i, args.logger)
            
            # self.Budget.append(time.time() - s_t)
            # args.logger.info('-'*50, self.Budget[-1])

        # for client in self.selected_clients:
        #         client.save_models()

    def set_clients(self, args, clientObj):
        for client in self.num_clients:
            # train_data = read_client_data(client, is_train=True)
            # test_data = read_client_data(client, is_train=False)
            client = clientObj(args, 
                            id=client, 
                            train_samples=8859, 
                            cuda_id=self.cuda_id[str(client)])
            self.clients.append(client)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self, round):
        assert (len(self.clients) > 0)

        # for client in self.clients:
        #     client.local_initialization(self.global_model)
        threads = [Thread(target=client.local_initialization, args=(self.global_model, self.writer, round, ))
                for client in self.selected_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_train_samples = 0
        for client in self.selected_clients:
            active_train_samples += client.train_samples

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples / active_train_samples)
            self.uploaded_ids.append(client.id)
            self.uploaded_models.append(client.model_ema)

    def add_parameters(self, w, client_model):
        client_model = client_model.to('cpu' if self.cuda_id["server"]==-1 else f'cuda:{self.cuda_id["server"]}')
        # for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
        #     server_param.data += client_param.data.clone() * w
        for (server_param_name, server_param), (_, client_param) in zip(self.global_model.named_parameters(), client_model.named_parameters()):
            if 'ridge' not in server_param_name:
                server_param.data += client_param.data.clone() * w
        #     # server_param.data += client_param.data.clone()
        #     # server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        self.global_model.to('cpu' if self.cuda_id["server"]==-1 else f'cuda:{self.cuda_id["server"]}')
        for param in self.global_model.parameters():
            param.data = torch.zeros_like(param.data)
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)


    def prepare_CLIP(self, args, device):
        # Prepare CLIP
        clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
        clip_size = clip_sizes[args.clip_variant]

        print("Using hidden layer CLIP space (Versatile Diffusion)")
        if not args.norm_embs:
            print("WARNING: YOU WANT NORMED EMBEDDINGS FOR VERSATILE DIFFUSION!")
        clip_extractor = Clipper(args.clip_variant, device=device, hidden_state=True, norm_embs=True)

        out_dim_image = 257 * clip_size # 257*768 = 197376
        out_dim_text  = 77  * clip_size # 77*768  = 59136

        print("clip_extractor loaded.")
        print("out_dim_image:",out_dim_image)
        print("out_dim_text:", out_dim_text)

        return clip_extractor
    
    def prepare_coco(self, data_path):
        # Preload coco captions
        nsda = NSDAccess(data_path)
        coco_73k = list(range(0, 73000))
        prompts_list = nsda.read_image_coco_info(coco_73k, info_type='captions')

        print("coco captions loaded.")

        return prompts_list

