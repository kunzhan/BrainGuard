import copy
import torch
import torch.nn as nn
from utils.DFL import DFL
from utils.loss_avg import AverageMeter
from utils.utils import soft_clip_loss, Clipper
import math
from trainmodel.models import *
import utils.data as data
from utils.utils import prepare_coco

class Client(object):
    def __init__(self, args, id, train_samples, cuda_id):

        self.cuda_id = cuda_id
        self.model = copy.deepcopy(args.model)
        self.model.ridge = RidgeRegression(input_size=args.multi_voxel_dims[id], out_features=2048)
        self.model = self.model.to('cuda:{}'.format(self.cuda_id))
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters(): # freeze the ema model parameters
            param.detach_()
        self.dataset = args.dataset
        self.device = 'cuda:{}'.format(self.cuda_id)
        self.id = id
        self.args = args
        self.train_samples = train_samples
        self.batch_size = args.batch_size
        self.global_rounds = args.global_rounds
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.set_opt_grouped_parameters(args)
        self.loss_mse = nn.MSELoss(reduction='mean').to(f'cuda:{self.cuda_id}')
        self.optimizer = torch.optim.AdamW(self.opt_grouped_parameters, betas=(0.9, 0.9999), lr=self.learning_rate, eps=1e-8)
        self.prompts_list = prepare_coco(args.data_root)
        self.clip_extractor = self.prepare_CLIP(args, self.device)

        self.prepare_dataloader()
        self.set_lr_scheduler(args)

        self.eta = args.eta
        self.layer_idx = args.layer_idx
        self.DFL = DFL(self.id, self.cuda_id, soft_clip_loss, self.train_dl, self.layer_idx, self.eta, self.device)
        
        self.global_best_val_sim_image= 0.
        self.global_model_best_val_sim_image= 0.
        self.all_steps = 0
        self.best_val_bwd = 0.
        self.flag_ala = True
        self.before_aggregate_bwd = 0.

        self.total_loss = AverageMeter()
        self.mse_image = AverageMeter()
        self.mse_text = AverageMeter()
        self.nce_image = AverageMeter()
        self.nce_text = AverageMeter()
        
    def train(self, writer, round, logger):

        self.model.to(f'cuda:{self.cuda_id}')
        self.model_ema.to(f'cuda:{self.cuda_id}')
        self.model.train()

        for step in range(self.local_steps):

            logger.info("Start train Client {}, global_round: {}/{}  Local step:{}/{}".format(self.id, round+1, self.global_rounds, step+1, self.local_steps))

            for train_i, data_i in enumerate(self.train_dl):
                self.train_i = train_i
                repeat_index = train_i % 3 # randomly choose the one in the repeated three
                voxel, image, coco = data_i
                voxel = voxel[:,repeat_index,...].float()

                coco_ids = coco.squeeze().tolist()
                current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
                captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

                if self.args.use_image_aug:
                    image = data.img_augment(image)

                clip_image = self.clip_extractor.embed_image(image).float()
                clip_text = self.clip_extractor.embed_text(captions).float()

                voxel = voxel.to(f'cuda:{self.cuda_id}')
                clip_image = clip_image.to(f'cuda:{self.cuda_id}')
                clip_text = clip_text.to(f'cuda:{self.cuda_id}')
                
                ridge_out = self.model.ridge(voxel)
                results = self.model.backbone(ridge_out)

                clip_image_pred = results[0]
                clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                loss_clip_image = soft_clip_loss(
                    clip_image_pred_norm,
                    clip_image_norm,
                    )
                
                clip_text_pred = results[1]
                clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
                clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)

                loss_mse_text = self.loss_mse(clip_text_pred_norm, clip_text_norm) * 10000

                loss_clip_text = soft_clip_loss(
                    clip_text_pred_norm,
                    clip_text_norm,
                    )
 
                loss =  loss_mse_image * 2 + loss_clip_image + loss_clip_text + loss_mse_text * 2
                self.update_local_ema(self.model, self.model_ema, self.args.ema_decay)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                current_lr = self.lr_scheduler.get_last_lr()[0]
                self.total_loss.update(loss.item())
                self.mse_image.update(loss_mse_image.item())
                self.mse_text.update(loss_mse_text.item())
                self.nce_image.update(loss_clip_image.item())
                self.nce_text.update(loss_clip_text.item())

                writer.add_scalar(f'Loss/loss_All_train_client_{self.id}', self.total_loss.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_Mse_image_train_client_{self.id}', self.mse_image.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_Mse_text_train_client_{self.id}', self.mse_text.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_SoftCliptrain_image_client_{self.id}', self.nce_image.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Loss/loss_SoftCliptrain_text_client_{self.id}', self.nce_text.avg, self.all_steps * len(self.train_dl) + train_i)
                writer.add_scalar(f'Learning rate/train_client_{self.id}', current_lr, self.all_steps * len(self.train_dl) + train_i)

                if (train_i % (len(self.train_dl) // 8) == 0):
                    logger.info(f"client{self.id}: Learning rate: {current_lr:.4f}, Loss softclip image:{self.nce_image.avg:.4f}, Loss softclip text:{self.nce_text.avg:.4f}")

            self.all_steps += 1
            self.model.eval()

            with torch.no_grad():
                val_sims_base_image = AverageMeter()
                val_sims_base_text = AverageMeter()
                val_loss_base_mse_image = AverageMeter()
                val_loss_base_mse_text = AverageMeter()
                val_loss_base_nce_image = AverageMeter()
                val_loss_base_nce_text = AverageMeter()

                for val_i, data_i in enumerate(self.val_dl):
                    self.val_i = val_i
                    repeat_index = val_i % 3
                    voxel, image, coco = data_i
                    voxel = torch.mean(voxel,axis=1)
                    voxel = voxel.to(f'cuda:{self.cuda_id}').float()

                    coco_ids = coco.squeeze().tolist()
                    current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
                    captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

                    clip_image = self.clip_extractor.embed_image(image).float()
                    clip_text = self.clip_extractor.embed_text(captions).float()

                    clip_image = clip_image.to(f'cuda:{self.cuda_id}')
                    clip_text = clip_text.to(f'cuda:{self.cuda_id}')

                    ridge_out = self.model.ridge(voxel)
                    results = self.model.backbone(ridge_out)

                    clip_image_pred = results[0]
                    clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                    clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                    val_loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                    loss_clip_image = soft_clip_loss(
                        clip_image_pred_norm,
                        clip_image_norm,
                    )

                    val_sims_image = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()

                    clip_text_pred = results[1]
                    clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
                    clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)

                    val_loss_mse_text = self.loss_mse(clip_text_pred_norm, clip_text_norm) * 10000

                    loss_clip_text = soft_clip_loss(
                        clip_text_pred_norm,
                        clip_text_norm,
                        )

                    val_sims_text = nn.functional.cosine_similarity(clip_text_norm, clip_text_pred_norm).mean().item()

                    val_loss_base_nce_image.update(loss_clip_image.item())
                    val_loss_base_nce_text.update(loss_clip_text.item())
                    val_loss_base_mse_image.update(val_loss_mse_image.item())
                    val_loss_base_mse_text.update(val_loss_mse_text.item())
                    val_sims_base_image.update(val_sims_image)
                    val_sims_base_text.update(val_sims_text)

                writer.add_scalar(f'Val/sim_image_{self.id}', val_sims_base_image.avg, self.all_steps)
                writer.add_scalar(f'Val/sim_text_{self.id}', val_sims_base_text.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_mse_image{self.id}', val_loss_base_mse_image.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_mse_text{self.id}', val_loss_base_mse_text.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_SoftClip_image{self.id}', val_loss_base_nce_image.avg, self.all_steps)
                writer.add_scalar(f'Val/loss_SoftClip_text{self.id}', val_loss_base_nce_text.avg, self.all_steps)

                logger.info(f'client{self.id}  Mean sim image: {val_sims_base_image.avg}, Mean sim text: {val_sims_base_text.avg}')
                if val_sims_base_image.avg > self.global_best_val_sim_image:
                    self.global_best_val_sim_image = val_sims_base_image.avg
                    torch.save(self.model.state_dict(), './logs/model/client{}_best.pth'.format(self.id))
            
        logger.info("Train Client {} done".format(self.id))


    def eval_global_model(self, global_model, writer, round, logger):
        
        global_sims_base_image = AverageMeter()
        global_sims_base_text = AverageMeter()
        global_loss_base_nce_image = AverageMeter()
        global_loss_base_nce_text = AverageMeter()
        global_loss_base_mse_image = AverageMeter()
        global_loss_base_mse_text = AverageMeter()
        
        self.model.ridge.to(f'cuda:{self.cuda_id}')
        global_model.backbone.to(f'cuda:{self.cuda_id}')

        self.model.eval()
        global_model.eval()

        with torch.no_grad():
            for val_i, data_i in enumerate(self.val_dl):
                self.val_i = val_i
                repeat_index = val_i % 3
                voxel, image, coco = data_i
                voxel = torch.mean(voxel,axis=1)

                coco_ids = coco.squeeze().tolist()
                current_prompts_list = [self.prompts_list[coco_id] for coco_id in coco_ids]
                captions = [prompts[repeat_index]['caption'] for prompts in current_prompts_list]

                clip_image = self.clip_extractor.embed_image(image).float()
                clip_text = self.clip_extractor.embed_text(captions).float()

                voxel = voxel.to(f'cuda:{self.cuda_id}').float()
                clip_text = clip_text.to(f'cuda:{self.cuda_id}').float()
                clip_image = clip_image.to(f'cuda:{self.cuda_id}')

                ridge_out = self.model.ridge(voxel)
                results = global_model.backbone(ridge_out)

                clip_image_pred = results[0]
                clip_image_pred_norm = nn.functional.normalize(clip_image_pred.flatten(1), dim=-1)
                clip_image_norm = nn.functional.normalize(clip_image.flatten(1), dim=-1)

                global_sims_image = nn.functional.cosine_similarity(clip_image_norm, clip_image_pred_norm).mean().item()
                loss_clip_image = soft_clip_loss(
                        clip_image_pred_norm,
                        clip_image_norm,
                        )
                global_loss_mse_image = self.loss_mse(clip_image_pred_norm, clip_image_norm) * 10000

                clip_text_pred = results[1]
                clip_text_pred_norm = nn.functional.normalize(clip_text_pred.flatten(1), dim=-1)
                clip_text_norm = nn.functional.normalize(clip_text.flatten(1), dim=-1)
                global_sims_text = nn.functional.cosine_similarity(clip_text_pred_norm, clip_text_norm).mean().item()

                global_loss_mse_text = self.loss_mse(clip_text_pred_norm, clip_text_norm) * 10000

                loss_clip_text = soft_clip_loss(
                        clip_text_pred_norm,
                        clip_text_norm,
                        )

                global_sims_base_image.update(global_sims_image)
                global_sims_base_text.update(global_sims_text)
                global_loss_base_nce_image.update(loss_clip_image.item())
                global_loss_base_nce_text.update(loss_clip_text.item())
                global_loss_base_mse_image.update(global_loss_mse_image.item())
                global_loss_base_mse_text.update(global_loss_mse_text.item())

            writer.add_scalar(f'Global_val/sim_image{self.id}', global_sims_base_image.avg, self.all_steps)
            writer.add_scalar(f'Global_val/sim_text{self.id}', global_sims_base_text.avg, self.all_steps)
            writer.add_scalar(f'Global_val/loss_mse_image{self.id}', global_loss_base_mse_image.avg, self.all_steps)
            writer.add_scalar(f'Global_val/loss_mse_text{self.id}', global_loss_base_mse_text.avg, self.all_steps)
            writer.add_scalar(f'Global_val/loss_nce_image{self.id}', global_loss_base_nce_image.avg, self.all_steps)
            writer.add_scalar(f'Global_val/loss_nce_text{self.id}', global_loss_base_nce_text.avg, self.all_steps)

            logger.info(f'Globel model on client{self.id} data:\n sim_image:{global_sims_base_image.avg:.4f} sim_text:{global_sims_base_text.avg:.4f}')
            self.model.ridge.to(f'cpu')
            global_model.backbone.to(f'cpu')


    def local_initialization(self, received_global_model, writer, round):
        self.model.to(f'cpu')
        temp_global_model = copy.deepcopy(received_global_model)
        temp_global_model.to(f'cpu')
        if self.flag_dfl:
            self.DFL.adaptive_local_aggregation(temp_global_model, self.model, writer, round, self.clip_extractor, self.prompts_list)

    
    def set_opt_grouped_parameters(self, args):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        self.opt_grouped_parameters = [
        {'params': [p for n, p in self.model.ridge.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in self.model.ridge.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in self.model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in self.model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},]


    def set_lr_scheduler(self, args):
        total_steps=((args.global_rounds * self.local_steps) * math.ceil(8859 / args.batch_size))
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=self.learning_rate,
            total_steps=total_steps,
            final_div_factor=1000,
            last_epoch=-1, pct_start=2/(args.global_rounds * self.local_steps)
        )

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
    
    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dl, self.val_dl = data.get_dls(
            subject=self.id,
            data_path=self.args.data_root,
            batch_size=self.args.batch_size,
            val_batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pool_type='max',
            pool_num=8192,
            length=8859,
            seed=42,
        )
        self.num_batches = len(self.train_dl)

    def update_local_ema(self, local_model, ema_model, alpha):
        for param, ema_param in zip(local_model.parameters(), ema_model.parameters()):
            ema_param.data = alpha * param.data + (1 - alpha) * ema_param.data