import json
import torch
import argparse
import time
import numpy as np
import pprint
from utils.data_utils import count_params
from servers.server import Server
from trainmodel.models import *
import logging
from utils.utils import init_log

def run(args):

    args.multi_voxel_dims = {1:15724, 2:14278, 5:13039, 7:12682}
    args.client_name = ['subj1', 'subj2', 'subj5', 'subj7']
    
    for i in range(args.prev, args.times):
        args.logger.info("Creating server and clients ...")

        clip_emb_dim = 768
        hidden_dim = 2048

        model = BrainGuardModule()
        model.ridge = RidgeRegression(input_size=15724, out_features=hidden_dim)
        model.backbone = BrainNetwork(in_dim=hidden_dim, latent_size=clip_emb_dim, out_dim_image=257*768, out_dim_text=77*768, use_projector=True, train_type=args.train_type)   
        
        args.logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
        args.model = model.to('cpu' if args.cuda_id["server"]==-1 else f'cuda:{args.cuda_id["server"]}')           
        args.logger.info(args.model)

        server = Server(args)

        if args.resume:
            args.logger.info("Starting test directly!")
            server.test(resume=True)
            args.logger.info("Resume test done!")
            break
            
        args.logger.info(f'==========Strat train {args.train_type} model==========')
        args.logger.info(f"============= Running time: {i}th =============")

        server.train(args)

    args.logger.info("All done!")



if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-dev', "--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('-data', "--dataset", type=str, default="NSD")
    parser.add_argument('-lbs', "--batch_size", type=int, default=50)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=3e-4, help="Local learning rate")
    parser.add_argument("--lr_scheduler_type",type=str,default='cycle', choices=['cycle','linear'],)
    parser.add_argument('-gr', "--global_rounds", type=int, default=600)
    parser.add_argument('-ls', "--local_steps", type=int, default=1)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-c', "--clients", type=int, default=[1,2,5,7], help="Train clients")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-p', "--layer_idx", type=int, default=24)
    parser.add_argument('-ed', "--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--cuda_id", type=json.loads, default='{"server":-1, "1": 0, "2": 1, "5": 2, "7": 3}', help="")
    parser.add_argument('--data_root', type=str, default='datapath')
    parser.add_argument("--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"], help='OpenAI clip variant',)
    parser.add_argument("--num_workers",type=int, default=5, help="Number of workers in dataloader")
    parser.add_argument("--norm_embs",action=argparse.BooleanOptionalAction, default=True, help="Do l2-norming of CLIP embeddings",)
    parser.add_argument("--use_image_aug",action=argparse.BooleanOptionalAction, default=True, help="whether to use image augmentation",)
    args = parser.parse_args()

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    args.logger = logger
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    all_args = {**vars(args)}
    print('{}\n'.format(pprint.pformat(all_args)))
    run(args)