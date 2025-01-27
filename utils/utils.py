import logging
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.nsd_access import NSDAccess

logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss

def simple_nce(preds, targs, logit_scale=3.51):
    # preds = preds.to(torch.float16)
    # targs = targs.to(torch.float16)
    brain_clip = logit_scale * preds @ targs.T
    clip_brain = logit_scale * targs @ preds.T

    labels = torch.arange(brain_clip.shape[0]).to(brain_clip.device)
    loss =  F.cross_entropy(brain_clip, labels).to(brain_clip.device)
    #if bidirectional:
    loss2 = F.cross_entropy(clip_brain, labels).to(clip_brain.device)
    loss = (loss + loss2)/2
    return loss

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))


def soft_clip_loss(preds, targs, temp=0.05, eps=1e-10):

    clip_clip = (targs @ targs.T)/temp + eps
    brain_clip = (preds @ targs.T)/temp + eps
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def prepare_coco(data_path):
    # Preload coco captions
    nsda = NSDAccess(data_path)
    coco_73k = list(range(0, 73000))
    prompts_list = nsda.read_image_coco_info(coco_73k,info_type='captions')

    print("coco captions loaded.")

    return prompts_list



from torchvision import transforms
import numpy as np
import clip
import torch.nn as nn
import random


def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')



class Clipper(torch.nn.Module):
    def __init__(self, clip_variant, clamp_embs=False, norm_embs=False,
                 hidden_state=False, device=torch.device('cpu')):
        super().__init__()
        assert clip_variant in ("RN50", "ViT-L/14", "ViT-B/32", "RN50x64"), \
            "clip_variant must be one of RN50, ViT-L/14, ViT-B/32, RN50x64"
        print(clip_variant, device)
        
        if clip_variant=="ViT-L/14" and hidden_state:
            from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
            image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            image_encoder = image_encoder.to(device)
            for param in image_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.image_encoder = image_encoder

            text_encoder = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
            text_encoder = text_encoder.to(device)
            for param in text_encoder.parameters():
                param.requires_grad = False # dont need to calculate gradients
            self.text_encoder = text_encoder
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        elif hidden_state:
            raise Exception("hidden_state embeddings only works with ViT-L/14 right now")
        
        clip_model, preprocess = clip.load(clip_variant, device=device)
        clip_model.eval() # dont want to train model
        for param in clip_model.parameters():
            param.requires_grad = False # dont need to calculate gradients
            
        self.clip = clip_model
        self.clip_variant = clip_variant
        if clip_variant == "RN50x64":
            self.clip_size = (448,448)
        else:
            self.clip_size = (224,224)
            
        preproc = transforms.Compose([
            transforms.Resize(size=self.clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
            transforms.CenterCrop(size=self.clip_size),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])
        self.preprocess = preproc
        self.hidden_state = hidden_state
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073])
        self.std = np.array([0.26862954, 0.26130258, 0.27577711])
        self.normalize = transforms.Normalize(self.mean, self.std)
        self.denormalize = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
        self.clamp_embs = clamp_embs
        self.norm_embs = norm_embs
        self.device= device
        
        def versatile_normalize_embeddings(encoder_output):
            embeds = encoder_output.last_hidden_state
            embeds = image_encoder.vision_model.post_layernorm(embeds)
            embeds = image_encoder.visual_projection(embeds)
            return embeds
        self.versatile_normalize_embeddings = versatile_normalize_embeddings

    def resize_image(self, image):
        # note: antialias should be False if planning to use Pinkney's Image Variation SD model
        return transforms.Resize(self.clip_size, antialias=None)(image.to(self.device))

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        if self.hidden_state:
            # clip_emb = self.preprocess((image/1.5+.25).to(self.device)) # for some reason the /1.5+.25 prevents oversaturation
            clip_emb = self.preprocess((image).to(self.device))
            clip_emb = self.image_encoder(clip_emb)
            clip_emb = self.versatile_normalize_embeddings(clip_emb)
        else:
            clip_emb = self.preprocess(image.to(self.device))
            clip_emb = self.clip.encode_image(clip_emb)
        # input is now in CLIP space, but mind-reader preprint further processes embeddings:
        if self.clamp_embs:
            clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
        if self.norm_embs:
            if self.hidden_state:        
                # normalize all tokens by cls token's norm
                clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
            else:
                clip_emb = nn.functional.normalize(clip_emb, dim=-1)
        return clip_emb
    
    def embed_text(self, prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
        """

        def normalize_embeddings(encoder_output):
            embeds = self.text_encoder.text_projection(encoder_output.last_hidden_state)
            embeds_pooled = encoder_output.text_embeds
            embeds = embeds / torch.norm(embeds_pooled.unsqueeze(1), dim=-1, keepdim=True)
            return embeds

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="pt").input_ids
        with torch.no_grad():
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.device),
            )
        prompt_embeds = normalize_embeddings(prompt_embeds)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        # bs_embed, seq_len, _ = prompt_embeds.shape
        # prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        # prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def embed_curated_annotations(self, annots):
        for i,b in enumerate(annots):
            t = ''
            while t == '':
                rand = torch.randint(5,(1,1))[0][0]
                t = b[0,rand]
            if i==0:
                txt = np.array(t)
            else:
                txt = np.vstack((txt,t))
        txt = txt.flatten()
        return self.embed_text(txt)
    
def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x


def decode_latents(latents,vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    return image


def combine_with_memory(
        pred_embedding_vision, pred_embedding_text, 
        clip_vision_train, clip_text_train,
        clip_vision_train_norm, clip_text_train_norm,
        clip_image_target, clip_text_target,
        clip_image_target_norm, clip_text_target_norm
        ):
    pred_embedding_text = pred_embedding_text.to(clip_text_train_norm.device)
    clip_image_target_norm = clip_image_target_norm.to(clip_text_train_norm.device)
    pred_embedding_vision = pred_embedding_vision.to(clip_text_train_norm.device)
    clip_text_target_norm = clip_text_target_norm.to(clip_text_train_norm.device)
    alpha = 0.1
    pred_embedding_text_norm = nn.functional.normalize(pred_embedding_text.flatten(1), dim=-1)
    pred_embedding_vision_norm = nn.functional.normalize(pred_embedding_vision.flatten(1), dim=-1)
    
    similarity_text = batchwise_cosine_similarity(pred_embedding_text_norm, clip_text_train_norm)
    similarity_vision = batchwise_cosine_similarity(pred_embedding_vision_norm, clip_vision_train_norm)
    # similarity_text = clip_text_train_norm @ pred_embedding_text_norm.T
    # similarity_vision = clip_vision_train_norm @ pred_embedding_vision_norm.T
    
    # Target with train data
    similarity_text_tar_with_train = batchwise_cosine_similarity(clip_text_target_norm, clip_text_train_norm)
    similarity_vision_tar_with_train = batchwise_cosine_similarity(clip_image_target_norm, clip_vision_train_norm) 
    topk_index_text_tar_with_train = torch.topk(similarity_text_tar_with_train.flatten(), 1).indices
    topk_index_vision_tar_with_train = torch.topk(similarity_vision_tar_with_train.flatten(), 1).indices

    topk_index_text = torch.topk(similarity_text.flatten(), 1).indices
    topk_index_vision = torch.topk(similarity_vision.flatten(), 1).indices
    # similarity_vision_by_text = pred_embedding_vision_norm @ (nn.functional.normalize(clip_vision_train[topk_index_text].flatten(1), dim=-1)).T
    similarity_vision_by_text = batchwise_cosine_similarity(pred_embedding_vision_norm, (nn.functional.normalize(clip_vision_train[topk_index_text].flatten(1), dim=-1)))
    print('\n Alpha' , alpha)
    print('target 跟 train data 的最大相似度')
    print('Top indices text tar retrival train data:',topk_index_text_tar_with_train,'Top similarity_text retrival train data:', torch.topk(similarity_text_tar_with_train.flatten(), 1).values)
    print('Top indices vision tar retrival train data:',topk_index_vision_tar_with_train,'Top similarity_vision retrival train data', torch.topk(similarity_vision_tar_with_train.flatten(), 1).values)
    
    print('预测在train data 中 top 10')
    print('Top 10 index text:',torch.topk(similarity_text.flatten(), 10).indices,'Top 10 similarity_text:', torch.topk(similarity_text.flatten(), 10).values)  
    print('Top 10 index vision:',torch.topk(similarity_vision.flatten(), 10).indices,'Top 10 similarity_vision', torch.topk(similarity_vision.flatten(), 10).values)

    print('target 在train data 中 top 10')
    print('Top 10 index text:',torch.topk(similarity_text_tar_with_train.flatten(), 10).indices,'Top 10 similarity_text:', torch.topk(similarity_text_tar_with_train.flatten(), 10).values)  
    print('Top 10 index vision:',torch.topk(similarity_vision_tar_with_train.flatten(), 10).indices,'Top 10 similarity_vision', torch.topk(similarity_vision_tar_with_train.flatten(), 10).values)

    print('预测与train data 中的最大相似度')
    print('Top indices text:',topk_index_text,'Top similarity_text:', torch.topk(similarity_text.flatten(), 1).values)
    print('Top indices vision:',topk_index_vision,'Top similarity_vision', torch.topk(similarity_vision.flatten(), 1).values)
    print('Top similarity_vision_by_text',torch.topk(similarity_vision_by_text.flatten(), 1).values)

    

    # import ipdb;ipdb.set_trace()
    combined_brain_clip_text_embeddings = (1-alpha) * clip_text_train[topk_index_text] + alpha * pred_embedding_text
    combined_brain_clip_image_embeddings = (1-alpha) * clip_vision_train[topk_index_vision] + alpha * pred_embedding_vision
    combined_brain_clip_image_embeddings_by_text = (1-alpha) * clip_vision_train[topk_index_text] + alpha * pred_embedding_vision

    combined_brain_clip_text_embeddings_using_target = (1-alpha) * clip_text_train[topk_index_text_tar_with_train] + alpha * pred_embedding_text
    combined_brain_clip_image_embeddings_using_target = (1-alpha) * clip_vision_train[topk_index_vision_tar_with_train] + alpha * pred_embedding_vision
    combined_brain_clip_image_embeddings_by_text_using_target = (1-alpha) * clip_vision_train[topk_index_text_tar_with_train] + alpha * pred_embedding_vision


    retrivaled_embedding_text_norm = nn.functional.normalize(clip_text_train[topk_index_text].flatten(1), dim=-1)
    retrivaled_embedding_vision_norm = nn.functional.normalize(clip_vision_train[topk_index_vision].flatten(1), dim=-1)

    similarity_text_retrival = batchwise_cosine_similarity(retrivaled_embedding_text_norm, clip_text_target_norm)
    similarity_vision_retrival = batchwise_cosine_similarity(retrivaled_embedding_vision_norm, clip_image_target_norm)
    print('检索到的跟tar之间的相似度')
    print('Similarity_retrival_text_with_tar', similarity_text_retrival)
    print('Similarity_retrival_vision_with_tar', similarity_vision_retrival)

    combined_brain_clip_text_embeddings_norm = nn.functional.normalize(combined_brain_clip_text_embeddings.flatten(1), dim=-1)
    combined_brain_clip_image_embeddings_norm = nn.functional.normalize(combined_brain_clip_image_embeddings.flatten(1), dim=-1)
    combined_brain_clip_image_embeddings_by_text_norm = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text.flatten(1), dim=-1)


    combined_brain_clip_text_embeddings_norm_using_tar = nn.functional.normalize(combined_brain_clip_text_embeddings_using_target.flatten(1), dim=-1)
    combined_brain_clip_image_embeddings_norm_using_tar = nn.functional.normalize(combined_brain_clip_image_embeddings_using_target.flatten(1), dim=-1)
    combined_brain_clip_image_embeddings_by_text_norm_using_tar = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text_using_target.flatten(1), dim=-1)

    similarity_text_after_using_tar = batchwise_cosine_similarity(clip_text_target_norm , combined_brain_clip_text_embeddings_norm_using_tar)
    similarity_vision_after_using_tar = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_norm_using_tar)
    similarity_vision_after_by_text_using_tar = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_norm_using_tar)
    print('使用 target 检索 进行合并后与 target 的相似度：')
    print('Similarity_text_combined_with_tar_using_tar:', torch.topk(similarity_text_after_using_tar.flatten(), 1).values)
    print('Similarity_vision_combined_with_combined_using_tar', torch.topk(similarity_vision_after_using_tar.flatten(), 1).values)
    print('Similarity_vision_combined_with_combined_by_text_using_tar', torch.topk(similarity_vision_after_by_text_using_tar.flatten(), 1).values)

    # print('Similarity_text_combined_with_tar:', torch.topk(similarity_text_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined', torch.topk(similarity_vision_after.flatten(), 1).values)
    # print('Similarity_vision_combined_with_combined_by_text', torch.topk(similarity_vision_after_by_text.flatten(), 1).values)
    # similarity_text_before = pred_embedding_text_norm @ clip_text_target_norm.T
    # similarity_vision_before = pred_embedding_vision_norm @ clip_image_target_norm.T

    similarity_text_before = batchwise_cosine_similarity(pred_embedding_text_norm, clip_text_target_norm)
    similarity_vision_before = batchwise_cosine_similarity(pred_embedding_vision_norm, clip_image_target_norm)

    # similarity_text_after = clip_text_target_norm @ combined_brain_clip_text_embeddings_norm.T
    # similarity_vision_after = clip_image_target_norm @ combined_brain_clip_image_embeddings_norm.T
    # similarity_vision_after_by_text = clip_image_target_norm @ combined_brain_clip_image_embeddings_by_text_norm.T

    similarity_text_after = batchwise_cosine_similarity(clip_text_target_norm , combined_brain_clip_text_embeddings_norm)
    similarity_vision_after = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_norm)
    similarity_vision_after_by_text = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_norm)
    print('pred 跟 target 相似度')
    print('Similarity_text_pred_with_tar:', torch.topk(similarity_text_before.flatten(), 1).values)
    print('Similarity_vision_pred_with_tar', torch.topk(similarity_vision_before.flatten(), 1).values)
    print('pred 自己检索合并后 跟 target 相似度')
    print('Similarity_text_combined_with_tar:', torch.topk(similarity_text_after.flatten(), 1).values)
    print('Similarity_vision_combined_with_tar', torch.topk(similarity_vision_after.flatten(), 1).values)
    print('Similarity_vision_combined_with_tar_by_text', torch.topk(similarity_vision_after_by_text.flatten(), 1).values)

    # similarity_text_after_combine = batchwise_cosine_similarity(clip_text_train_norm , combined_brain_clip_text_embeddings_norm)
    # topk_index_text_after_combine = torch.topk(similarity_text_after_combine.flatten(), 1).indices

    # print('Top indices text after combine:',topk_index_text_after_combine,'Top similarity_text_after_combine:', torch.topk(similarity_text_after_combine.flatten(), 1).values)
    # combined_brain_clip_image_embeddings_by_text_after_combine = (1-alpha) * clip_vision_train[topk_index_text] + alpha * pred_embedding_vision
    # combined_brain_clip_image_embeddings_by_text_after_combine_norm = nn.functional.normalize(combined_brain_clip_image_embeddings_by_text_after_combine.flatten(1), dim=-1)
    # similarity_vision_after_by_text_after_combine = batchwise_cosine_similarity(clip_image_target_norm , combined_brain_clip_image_embeddings_by_text_after_combine_norm)
    # print('Similarity_vision_combined_with_combined_by_text_after_combine', torch.topk(similarity_vision_after_by_text_after_combine.flatten(), 1).values)


    # return combined_brain_clip_image_embeddings, combined_brain_clip_text_embeddings
    return combined_brain_clip_image_embeddings_using_target, combined_brain_clip_text_embeddings_using_target

@torch.no_grad()
def reconstruction(
    args,
    image, voxel, captions, 
    clip_vision_train, clip_text_train,
    clip_vision_train_norm, clip_text_train_norm,
    voxel2clip,
    clip_extractor,
    unet, vae, noise_scheduler,
    img_lowlevel = None,
    num_inference_steps = 50,
    recons_per_sample = 1,
    guidance_scale = 7.5,
    img2img_strength = .85,
    seed = 42,
    plotting=True,
    verbose=False,
    n_samples_save=1,
    device = None,
    mem_efficient = True,
    retrival_from_memory = False,

):
    assert n_samples_save==1, "n_samples_save must = 1. Function must be called one image at a time"
    assert recons_per_sample>0, "recons_per_sample must > 0"
    
    brain_recons = None
    
    voxel=voxel[:n_samples_save]
    image=image[:n_samples_save]
    B = voxel.shape[0]

    clip_image_target = clip_extractor.embed_image(image)
    clip_text_target = clip_extractor.embed_text(captions)

    clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim=-1)
    clip_text_target_norm = nn.functional.normalize(clip_text_target.flatten(1), dim=-1)

    if mem_efficient:
        clip_extractor.to("cpu")
        unet.to("cpu")
        vae.to("cpu")
    else:
        clip_extractor.to(device)
        unet.to(device)
        vae.to(device)

    if unet is not None:
        do_classifier_free_guidance = guidance_scale > 1.0
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = unet.config.sample_size * vae_scale_factor
        width = unet.config.sample_size * vae_scale_factor
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    if voxel2clip is not None:
        ridge_out = voxel2clip.ridge(voxel)
        clip_results = voxel2clip.backbone(ridge_out)
        if mem_efficient:
            voxel2clip.to('cpu')
        # brain_clip_text_embeddings = clip_extractor.embed_text(captions).float()
        brain_clip_image_embeddings, brain_clip_text_embeddings = clip_results[:2]
        if retrival_from_memory:
            brain_clip_image_embeddings, brain_clip_text_embeddings = combine_with_memory(
                brain_clip_image_embeddings, brain_clip_text_embeddings, 
                clip_vision_train, clip_text_train, 
                clip_vision_train_norm, clip_text_train_norm,
                clip_image_target, clip_text_target,
                clip_image_target_norm, clip_text_target_norm)
        # import ipdb;ipdb.set_trace()
        brain_clip_image_embeddings = brain_clip_image_embeddings.reshape(B,-1,768)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.reshape(B,-1,768)

        brain_clip_image_embeddings = brain_clip_image_embeddings.repeat(recons_per_sample, 1, 1)
        brain_clip_text_embeddings  = brain_clip_text_embeddings.repeat(recons_per_sample, 1, 1)

    if recons_per_sample > 0:
        for samp in range(len(brain_clip_image_embeddings)):
            brain_clip_image_embeddings[samp] = brain_clip_image_embeddings[samp]/(brain_clip_image_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
            brain_clip_text_embeddings[samp]  = brain_clip_text_embeddings[samp]/(brain_clip_text_embeddings[samp,0].norm(dim=-1).reshape(-1, 1, 1) + 1e-6)
        input_embedding = brain_clip_image_embeddings#.repeat(recons_per_sample, 1, 1)
        if verbose: print("input_embedding",input_embedding.shape)

        prompt_embeds = brain_clip_text_embeddings
        if verbose: print("prompt_embedding",prompt_embeds.shape)

        if do_classifier_free_guidance:
            input_embedding = torch.cat([torch.zeros_like(input_embedding), input_embedding]).to(device).to(unet.dtype)
            prompt_embeds = torch.cat([torch.zeros_like(prompt_embeds), prompt_embeds]).to(device).to(unet.dtype)

        # 3. dual_prompt_embeddings
        input_embedding = torch.cat([prompt_embeds, input_embedding], dim=1)

        # 4. Prepare timesteps
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

        # 5b. Prepare latent variables
        batch_size = input_embedding.shape[0] // 2 # divide by 2 bc we doubled it for classifier-free guidance
        shape = (batch_size, unet.in_channels, height // vae_scale_factor, width // vae_scale_factor)
        if img_lowlevel is not None: # use img_lowlevel for img2img initialization
            init_timestep = min(int(num_inference_steps * img2img_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = noise_scheduler.timesteps[t_start:]
            latent_timestep = timesteps[:1].repeat(batch_size)
            
            if verbose: print("img_lowlevel", img_lowlevel.shape)
            img_lowlevel_embeddings = clip_extractor.normalize(img_lowlevel)
            if verbose: print("img_lowlevel_embeddings", img_lowlevel_embeddings.shape)
            if mem_efficient:
                vae.to(device)
            init_latents = vae.encode(img_lowlevel_embeddings.to(device).to(vae.dtype)).latent_dist.sample(generator)
            init_latents = vae.config.scaling_factor * init_latents
            init_latents = init_latents.repeat(recons_per_sample, 1, 1, 1)

            noise = torch.randn([recons_per_sample, 4, 64, 64], device=device, 
                                generator=generator, dtype=input_embedding.dtype)
            init_latents = noise_scheduler.add_noise(init_latents, noise, latent_timestep)
            latents = init_latents
        else:
            timesteps = noise_scheduler.timesteps
            latents = torch.randn([recons_per_sample, 4, 64, 64], device=device,
                                  generator=generator, dtype=input_embedding.dtype)
            latents = latents * noise_scheduler.init_noise_sigma

        # 7. Denoising loop
        if mem_efficient:
            unet.to(device)
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            if verbose: print("timesteps: {}, latent_model_input: {}, input_embedding: {}".format(i, latent_model_input.shape, input_embedding.shape))
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embedding).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
        
        if mem_efficient:
            unet.to("cpu")

        recons = decode_latents(latents.to(device),vae.to(device)).detach().cpu()

        brain_recons = recons.unsqueeze(0)

    if verbose: print("brain_recons",brain_recons.shape)
                    
    # pick best reconstruction out of several
    best_picks = np.zeros(n_samples_save).astype(np.int16)
    
    if mem_efficient:
        vae.to("cpu")
        unet.to("cpu")
        clip_extractor.to(device)

    # clip_image_target = clip_extractor.embed_image(image)
    # clip_image_target_norm = nn.functional.normalize(clip_image_target.flatten(1), dim=-1)
    sims=[]
    for im in range(recons_per_sample): 
        currecon = clip_extractor.embed_image(brain_recons[0,[im]].float()).to(clip_image_target_norm.device).to(clip_image_target_norm.dtype)
        currecon = nn.functional.normalize(currecon.view(len(currecon),-1),dim=-1)
        # import ipdb;ipdb.set_trace()
        cursim = batchwise_cosine_similarity(clip_image_target_norm,currecon)
        sims.append(cursim.item())
    if verbose: print(sims)
    best_picks[0] = int(np.nanargmax(sims))   
    if verbose: print(best_picks)
    if mem_efficient:
        clip_extractor.to("cpu")
        voxel2clip.to(device)
                    
    img2img_samples = 0 if img_lowlevel is None else 1
    num_xaxis_subplots = 1+img2img_samples+recons_per_sample
    if plotting:
        fig, ax = plt.subplots(n_samples_save, num_xaxis_subplots, 
                           figsize=(num_xaxis_subplots*5,6*n_samples_save),facecolor=(1, 1, 1))
    else:
        fig = None
        recon_img = None
    
    im = 0
    if plotting:
        ax[0].set_title(f"Original Image")
        ax[0].imshow(torch_to_Image(image[im]))
        if img2img_samples == 1:
            ax[1].set_title(f"Img2img ({img2img_strength})")
            ax[1].imshow(torch_to_Image(img_lowlevel[im].clamp(0,1)))
    for ii,i in enumerate(range(num_xaxis_subplots-recons_per_sample,num_xaxis_subplots)):
        recon = brain_recons[im][ii]
        if plotting:
            if ii == best_picks[im]:
                ax[i].set_title(f"Reconstruction",fontweight='bold')
                recon_img = recon
            else:
                ax[i].set_title(f"Recon {ii+1} from brain")
            ax[i].imshow(torch_to_Image(recon))
    if plotting:
        for i in range(num_xaxis_subplots):
            ax[i].axis('off')
    
    return fig, brain_recons, best_picks, recon_img, ridge_out