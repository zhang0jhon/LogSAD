import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import resize


import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
from torchvision.ops.focal_loss import sigmoid_focal_loss
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from sklearn.mixture import GaussianMixture
import faiss
import open_clip_local as open_clip

from torch.utils.data.dataset import ConcatDataset
from scipy.optimize import linear_sum_assignment
from sklearn.random_projection import SparseRandomProjection
import cv2
from torchvision.transforms import InterpolationMode
from PIL import Image
import string

from prompt_ensemble import encode_text_with_prompt_ensemble, encode_normal_text, encode_abnormal_text, encode_general_text, encode_obj_text
from kmeans_pytorch import kmeans, kmeans_predict
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from matplotlib import pyplot as plt

import pickle
from scipy.stats import norm

from open_clip_local.pos_embed import get_2d_sincos_pos_embed

from anomalib.models.components import KCenterGreedy

def to_np_img(m):
    m = m.permute(1, 2, 0).cpu().numpy()
    mean = np.array([[[0.48145466, 0.4578275, 0.40821073]]])
    std = np.array([[[0.26862954, 0.26130258, 0.27577711]]])
    m  = m * std + mean
    return np.clip((m * 255.), 0, 255).astype(np.uint8)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MyModel(nn.Module):
    """Example model class for track 2.

    This class applies few-shot anomaly detection using the WinClip model from Anomalib.
    """

    def __init__(self) -> None:
        super().__init__()

        setup_seed(42)
        # NOTE: Create your transformation pipeline (if needed).
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.transform = v2.Compose(
            [
                v2.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
            ],
        )

        # NOTE: Create your model.
       
        self.model_clip, _, _ = open_clip.create_model_and_transforms('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K')
        self.feature_list = [6, 12, 18, 24]
        self.embed_dim = 768
        self.vision_width = 1024

        self.model_sam = sam_model_registry["vit_h"](checkpoint = "./checkpoint/sam_vit_h_4b8939.pth").to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(model = self.model_sam)

        self.memory_size = 2048
        self.n_neighbors = 2

        self.model_clip.eval()
        self.test_args = None
        self.align_corners = True # False
        self.antialias = True # False
        self.inter_mode = 'bilinear' # bilinear/bicubic 
        
        self.cluster_feature_id = [0, 1]

        self.cluster_num_dict = {
            "breakfast_box": 3, # unused
            "juice_bottle": 8, # unused
            "splicing_connectors": 10, # unused
            "pushpins": 10, 
            "screw_bag": 10,
        }
        self.query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box', 'black background'],
            "juice_bottle": ['bottle', ['black background', 'background']],
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['screw'], 'plastic bag', 'background'],
            "splicing_connectors": [['splicing connector', 'splice connector',], ['cable', 'wire'], ['grid']],
        }
        self.foreground_label_idx = {  # for query_words_dict
            "breakfast_box": [0, 1, 2, 3, 4, 5],
            "juice_bottle": [0],
            "pushpins": [0],
            "screw_bag": [0], 
            "splicing_connectors":[0, 1]
        }

        self.patch_query_words_dict = {
            "breakfast_box": ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box', 'black background'],
            "juice_bottle": [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'], ['black background', 'background']], 
            "pushpins": [['pushpin', 'pin'], ['plastic box', 'black background']],
            "screw_bag": [['hex screw', 'hexagon bolt'], ['hex nut', 'hexagon nut'], ['ring washer', 'ring gasket'], ['plastic bag', 'background']], # 79.71
            "splicing_connectors": [['splicing connector', 'splice connector',], ['cable', 'wire'], ['grid']],
        }
        

        self.query_threshold_dict = {
            "breakfast_box": [0., 0., 0., 0., 0., 0., 0.], # unused
            "juice_bottle": [0., 0., 0.], # unused
            "splicing_connectors": [0.15, 0.15, 0.15, 0., 0.], # unused
            "pushpins": [0.2, 0., 0., 0.],
            "screw_bag": [0., 0., 0.,],
        }

        self.feat_size = 64
        self.ori_feat_size = 32

        self.visualization = False #False # True #False

        self.pushpins_count = 15

        self.splicing_connectors_count = [2, 3, 5] # coresponding to yellow, blue, and red
        self.splicing_connectors_distance = 0
        self.splicing_connectors_cable_color_query_words_dict = [['yellow cable', 'yellow wire'], ['blue cable', 'blue wire'], ['red cable', 'red wire']]
        
        self.juice_bottle_liquid_query_words_dict = [['red liquid', 'cherry juice'], ['yellow liquid', 'orange juice'], ['milky liquid']]
        self.juice_bottle_fruit_query_words_dict = ['cherry', ['tangerine', 'orange'], 'banana'] 

        # query words
        self.foreground_pixel_hist = 0  
        self.foreground_pixel_hist_screw_bag = 366.0 # 4-shot statistics
        self.foreground_pixel_hist_splicing_connectors = 4249.666666666667  # 4-shot statistics
        # patch query words
        self.patch_token_hist = []

        self.few_shot_inited = False

        self.save_coreset_features = False


        from dinov2.dinov2.hub.backbones import dinov2_vitl14
        self.model_dinov2 = dinov2_vitl14()
        self.model_dinov2.to(self.device)
        self.model_dinov2.eval()
        self.feature_list_dinov2 = [6, 12, 18, 24]
        self.vision_width_dinov2 = 1024

        self.stats = pickle.load(open("memory_bank/statistic_scores_model_ensemble_val.pkl", "rb"))

        self.mem_instance_masks = None

        self.anomaly_flag = False
        self.validation = False #True #False

    def set_save_coreset_features(self, save_coreset_features):
        self.save_coreset_features = save_coreset_features

    def set_viz(self, viz):
        self.visualization = viz

    def set_val(self, val):
        self.validation = val

    def forward(self, batch: torch.Tensor, batch_path: list) -> dict[str, torch.Tensor]:
        """Transform the input batch and pass it through the model.

        This model returns a dictionary with the following keys
        - ``anomaly_map`` - Anomaly map.
        - ``pred_score`` - Predicted anomaly score.
        """
        self.anomaly_flag = False
        batch = self.transform(batch).to(self.device)
        results = self.forward_one_sample(batch, self.mem_patch_feature_clip_coreset, self.mem_patch_feature_dinov2_coreset, batch_path[0])

        hist_score = results['hist_score']
        structural_score = results['structural_score']
        instance_hungarian_match_score = results['instance_hungarian_match_score']


        if self.validation:
            return {"hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score), "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}

        def sigmoid(z):
            return 1/(1 + np.exp(-z))
        
        # standardization
        standard_structural_score = (structural_score - self.stats[self.class_name]["structural_scores"]["mean"]) / self.stats[self.class_name]["structural_scores"]["unbiased_std"]
        standard_instance_hungarian_match_score = (instance_hungarian_match_score - self.stats[self.class_name]["instance_hungarian_match_scores"]["mean"]) / self.stats[self.class_name]["instance_hungarian_match_scores"]["unbiased_std"]
 
        pred_score = max(standard_instance_hungarian_match_score, standard_structural_score)
        pred_score = sigmoid(pred_score)


        if self.anomaly_flag:
            pred_score = 1.
            self.anomaly_flag = False


        return {"pred_score": torch.tensor(pred_score), "hist_score": torch.tensor(hist_score), "structural_score": torch.tensor(structural_score), "instance_hungarian_match_score": torch.tensor(instance_hungarian_match_score)}
        

    def forward_one_sample(self, batch: torch.Tensor, mem_patch_feature_clip_coreset: torch.Tensor, mem_patch_feature_dinov2_coreset: torch.Tensor, path: str):

        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(batch, self.feature_list)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (1, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (1, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1) # (1x64x64, 1024x4)
        
        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(batch, out_layer_list=self.feature_list)
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (1, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1) # (1x64x64, 1024x4)
        
        '''adding for kmeans seg '''
        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(self.feat_size * self.feat_size, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        mid_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            mid_features = temp_feat if mid_features is None else torch.cat((mid_features, temp_feat), -1)
            
        if self.feat_size != self.ori_feat_size:
            mid_features = mid_features.view(1, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            mid_features = F.interpolate(mid_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            mid_features = mid_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        mid_features = F.normalize(mid_features, p=2, dim=-1)
             
        results = self.histogram(batch, mid_features, proj_patch_tokens, self.class_name, os.path.dirname(path).split('/')[-1] + "_" + os.path.basename(path).split('.')[0])
        
        hist_score = results['score']

        '''calculate patchcore'''
        anomaly_maps_patchcore = []

        if self.class_name in ['pushpins', 'screw_bag']: # clip feature for patchcore
            len_feature_list = len(self.feature_list)
            for patch_feature, mem_patch_feature in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), mem_patch_feature_clip_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        if self.class_name in ['splicing_connectors', 'breakfast_box', 'juice_bottle']: # dinov2 feature for patchcore
            len_feature_list = len(self.feature_list_dinov2)
            for patch_feature, mem_patch_feature in zip(patch_tokens_dinov2.chunk(len_feature_list, dim=-1), mem_patch_feature_dinov2_coreset.chunk(len_feature_list, dim=-1)):
                patch_feature = F.normalize(patch_feature, dim=-1)
                mem_patch_feature = F.normalize(mem_patch_feature, dim=-1)
                normal_map_patchcore = (patch_feature @ mem_patch_feature.T)
                normal_map_patchcore = (normal_map_patchcore.max(1)[0]).cpu().numpy() # 1: normal 0: abnormal  
                anomaly_map_patchcore = 1 - normal_map_patchcore 

                anomaly_maps_patchcore.append(anomaly_map_patchcore)

        structural_score = np.stack(anomaly_maps_patchcore).mean(0).max()
        # anomaly_map_structural = np.stack(anomaly_maps_patchcore).mean(0).reshape(self.feat_size, self.feat_size)

        instance_masks = results["instance_masks"] 
        anomaly_instances_hungarian = []
        instance_hungarian_match_score = 1.
        if self.mem_instance_masks is not None and len(instance_masks) != 0:
            for patch_feature, mem_instance_features_single_stage in zip(patch_tokens_clip.chunk(len_feature_list, dim=-1), self.mem_instance_features_multi_stage.chunk(len_feature_list, dim=1)):
                instance_features = [patch_feature[mask, :].mean(0, keepdim=True) for mask in instance_masks]
                instance_features = torch.cat(instance_features, dim=0)
                instance_features = F.normalize(instance_features, dim=-1)

                normal_instance_hungarian = (instance_features @ mem_instance_features_single_stage.T)
                cost_matrix = (1 - normal_instance_hungarian).cpu().numpy()
                
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                cost = cost_matrix[row_ind, col_ind].sum() 
                cost = cost / min(cost_matrix.shape)
                anomaly_instances_hungarian.append(cost)

            instance_hungarian_match_score = np.mean(anomaly_instances_hungarian) 

        results = {'hist_score': hist_score, 'structural_score': structural_score,  'instance_hungarian_match_score': instance_hungarian_match_score}
        
        return results


    def histogram(self, image, cluster_feature, proj_patch_token, class_name, path):
        def plot_results_only(sorted_anns):
            cur = 1
            img_color = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
            for ann in sorted_anns:
                m = ann['segmentation']
                img_color[m] = cur
                cur += 1
            return img_color
        
        def merge_segmentations(a, b, background_class):
            unique_labels_a = np.unique(a)
            unique_labels_b = np.unique(b)

            max_label_a = unique_labels_a.max()
            label_map = np.zeros(max_label_a + 1, dtype=int)

            for label_a in unique_labels_a:
                mask_a = (a == label_a)

                labels_b = b[mask_a]
                if labels_b.size > 0:
                    count_b = np.bincount(labels_b, minlength=unique_labels_b.max() + 1)
                    label_map[label_a] = np.argmax(count_b)
                else:
                    label_map[label_a] = background_class # default background

            merged_a = label_map[a]
            return merged_a
        
        pseudo_labels = kmeans_predict(cluster_feature, self.cluster_centers, 'euclidean', device=self.device)
        kmeans_mask = torch.ones_like(pseudo_labels) * (self.classes - 1)    # default to background
        
        for pl in pseudo_labels.unique():
            mask = (pseudo_labels == pl).reshape(-1)
            # filter small region
            binary = mask.cpu().numpy().reshape(self.feat_size, self.feat_size).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 8:
                    mask[temp_mask.reshape(-1)] = False

            if mask.any():
                region_feature = proj_patch_token[mask, :].mean(0, keepdim=True)
                similarity = (region_feature @ self.query_obj.T)
                prob, index = torch.max(similarity, dim=-1)
                temp_label = index.squeeze(0).item()
                temp_prob = prob.squeeze(0).item()
                if temp_prob > self.query_threshold_dict[class_name][temp_label]: # threshold for each class
                    kmeans_mask[mask] = temp_label    


        raw_image = to_np_img(image[0])
        height, width = raw_image.shape[:2]
        masks = self.mask_generator.generate(raw_image)
        # self.predictor.set_image(raw_image)
        
        kmeans_label = pseudo_labels.view(self.feat_size, self.feat_size).cpu().numpy()
        kmeans_mask = kmeans_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        patch_similarity = (proj_patch_token @ self.patch_query_obj.T)
        patch_mask = patch_similarity.argmax(-1) 
        patch_mask = patch_mask.view(self.feat_size, self.feat_size).cpu().numpy()

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        sam_mask = plot_results_only(sorted_masks).astype(np.int)
        
        resized_mask = cv2.resize(kmeans_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        merge_sam = merge_segmentations(sam_mask, resized_mask, background_class=self.classes-1)

        resized_patch_mask = cv2.resize(patch_mask, (width, height), interpolation = cv2.INTER_NEAREST)
        patch_merge_sam = merge_segmentations(sam_mask, resized_patch_mask, background_class=self.patch_query_obj.shape[0]-1)
        
        # filter small region for merge sam
        binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32: # 448x448 
                merge_sam[temp_mask] = self.classes - 1 # set to background

        # filter small region for patch merge sam
        binary = (patch_merge_sam != (self.patch_query_obj.shape[0]-1) ).astype(np.uint8)  # foreground 1  background 0
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, num_labels):
            temp_mask = labels == i
            if np.sum(temp_mask) <= 32: # 448x448
                patch_merge_sam[temp_mask] = self.patch_query_obj.shape[0]-1 # set to background

        score = 0. # default to normal
        self.anomaly_flag = False
        instance_masks = []
        if self.class_name == 'pushpins':
            # object count hist
            kernel = np.ones((3, 3), dtype=np.uint8)  # dilate for robustness 
            binary = np.isin(merge_sam, self.foreground_label_idx[self.class_name]).astype(np.uint8) # foreground 1  background 0
            dilate_binary = cv2.dilate(binary, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilate_binary, connectivity=8)
            pushpins_count = num_labels - 1 # number of pushpins

            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            if self.few_shot_inited and pushpins_count != self.pushpins_count and self.anomaly_flag is False:
                self.anomaly_flag = True
                print('number of pushpins: {}, but canonical number of pushpins: {}'.format(pushpins_count, self.pushpins_count))
            
            # patch hist 
            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            binary_foreground = dilate_binary.astype(np.uint8) 

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

                # plt.savefig('pic/vis/{}_seg_{}.png'.format(class_name, path), bbox_inches='tight', pad_inches = 0) # pad_inches = 0
                # plt.close()


            # todo: same number in total but in different boxes or broken box
            return {"score": score, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'splicing_connectors':
            #  object count hist for default
            sam_mask_max_area = sorted_masks[0]['segmentation'] # background
            binary = (sam_mask_max_area == 0).astype(np.uint8) # sam_mask_max_area is background,  background 0 foreground 1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            count = 0
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448 64
                    binary[temp_mask] = 0 # set to background
                else:
                    count += 1
            if count != 1 and self.anomaly_flag is False: # cable cut or no cable or no connector
                print('number of connected component in splicing_connectors: {}, but the default connected component is 1.'.format(count))
                self.anomaly_flag = True

            merge_sam[~(binary.astype(np.bool))] = self.query_obj.shape[0] - 1 # remove noise
            patch_merge_sam[~(binary.astype(np.bool))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            # erode the cable and divide into left and right parts
            kernel = np.ones((23, 23), dtype=np.uint8)
            erode_binary = cv2.erode(binary, kernel)
            h, w = erode_binary.shape
            distance = 0

            left, right = erode_binary[:, :int(w/2)],  erode_binary[:, int(w/2):]   
            left_count = np.bincount(left.reshape(-1), minlength=self.classes)[1]  # foreground
            right_count = np.bincount(right.reshape(-1), minlength=self.classes)[1] # foreground

            # binary_cable = (merge_sam == 1).astype(np.uint8)   
            binary_cable = (patch_merge_sam == 1).astype(np.uint8) 
            
            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_cable = cv2.erode(binary_cable, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_cable, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448
                    binary_cable[temp_mask] = 0 # set to background
                

            binary_cable = cv2.resize(binary_cable, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

            binary_clamps = (patch_merge_sam == 0).astype(np.uint8)

            kernel = np.ones((5, 5), dtype=np.uint8)
            binary_clamps = cv2.erode(binary_clamps, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_clamps, connectivity=8)
            for i in range(1, num_labels):
                temp_mask = labels == i
                if np.sum(temp_mask) <= 64: # 448x448
                    binary_clamps[temp_mask] = 0 # set to background
                else:
                    instance_mask = temp_mask.astype(np.uint8)
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                    if instance_mask.any():
                        instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            binary_clamps = cv2.resize(binary_clamps, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)

            binary_connector = cv2.resize(binary, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            
            query_cable_color = encode_obj_text(self.model_clip, self.splicing_connectors_cable_color_query_words_dict, self.tokenizer, self.device)
            cable_feature = proj_patch_token[binary_cable.astype(np.bool).reshape(-1), :].mean(0, keepdim=True)
            idx_color = (cable_feature @ query_cable_color.T).argmax(-1).squeeze(0).item()
            foreground_pixel_count = np.sum(erode_binary) / self.splicing_connectors_count[idx_color]


            slice_cable = binary[:, int(w/2)-1: int(w/2)+1]
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(slice_cable, connectivity=8)
            cable_count = num_labels - 1
            if cable_count != 1 and self.anomaly_flag is False: # two cables
                print('number of cable count in splicing_connectors: {}, but the default cable count is 1.'.format(cable_count))
                self.anomaly_flag = True

            # {2-clamp: yellow  3-clamp: blue  5-clamp: red}    cable color and clamp number mismatch
            if self.few_shot_inited and self.foreground_pixel_hist_splicing_connectors != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist_splicing_connectors
                if (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False:    # color and number mismatch
                    print('cable color and number of clamps mismatch, cable color idx: {} (0: yellow 2-clamp, 1: blue 3-clamp, 2: red 5-clamp), foreground_pixel_count :{}, canonical foreground_pixel_hist: {}.'.format(idx_color, foreground_pixel_count, self.foreground_pixel_hist_splicing_connectors))
                    self.anomaly_flag = True

            # left right hist for symmetry
            ratio = np.sum(left_count) / (np.sum(right_count) + 1e-5)
            if self.few_shot_inited and (ratio > 1.2 or ratio < 0.8) and self.anomaly_flag is False: # left right asymmetry in clamp
                print('left and right connectors are not symmetry.')
                self.anomaly_flag = True

            # left and right centroids distance
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(erode_binary, connectivity=8)
            if num_labels - 1 == 2:
                centroids = centroids[1:]
                x1, y1 = centroids[0] 
                x2, y2 = centroids[1]
                distance = np.sqrt((x1/w - x2/w)**2 + (y1/h - y2/h)**2)
                if self.few_shot_inited and self.splicing_connectors_distance != 0 and self.anomaly_flag is False:
                    ratio = distance / self.splicing_connectors_distance
                    if ratio < 0.6 or ratio > 1.4:  # too short or too long centroids distance (cable) # 0.6 1.4
                        print('cable is too short or too long.')
                        self.anomaly_flag = True

            # patch hist 
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])#[:-1]  # ignore background (grid) for statistic
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            # todo    mismatch cable link  
            binary_foreground = binary.astype(np.uint8) # only 1 instance, so additionally seperate cable and clamps
            if binary_connector.any():
                instance_masks.append(binary_connector.astype(np.bool).reshape(-1))
            if binary_clamps.any():
                instance_masks.append(binary_clamps.astype(np.bool).reshape(-1))
            if binary_cable.any():
                instance_masks.append(binary_cable.astype(np.bool).reshape(-1))      

            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, binary_connector, merge_sam, patch_merge_sam, erode_binary, binary_cable, binary_clamps]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'binary_connector', 'merge sam', 'patch merge sam', 'erode binary', 'binary_cable', 'binary_clamps']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()
                # plt.savefig('pic/{}/seg_{}.png'.format(class_name, path))
                # plt.close()

            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "distance": distance, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'screw_bag':
            # pixel hist of kmeans mask
            foreground_pixel_count = np.sum(np.bincount(kmeans_mask.reshape(-1))[:len(self.foreground_label_idx[self.class_name])])  # foreground pixel
            if self.few_shot_inited and self.foreground_pixel_hist_screw_bag != 0 and self.anomaly_flag is False:
                ratio = foreground_pixel_count / self.foreground_pixel_hist_screw_bag
                # todo: optimize
                if ratio < 0.94 or ratio > 1.06: # 82.95 |    81.3 
                    print('foreground pixel histagram of screw bag: {}, the canonical foreground pixel histogram of screw bag in few shot: {}'.format(foreground_pixel_count, self.foreground_pixel_hist_screw_bag))
                    self.anomaly_flag = True

            # patch hist
            binary_screw = np.isin(kmeans_mask, self.foreground_label_idx[self.class_name])
            patch_mask[~binary_screw] = self.patch_query_obj.shape[0] - 1 # remove patch noise
            resized_binary_screw = cv2.resize(binary_screw.astype(np.uint8), (patch_merge_sam.shape[1], patch_merge_sam.shape[0]), interpolation = cv2.INTER_NEAREST)
            patch_merge_sam[~(resized_binary_screw.astype(np.bool))] = self.patch_query_obj.shape[0] - 1 # remove patch noise

            clip_patch_hist = np.bincount(patch_mask.reshape(-1), minlength=self.patch_query_obj.shape[0])[:-1]
            clip_patch_hist = clip_patch_hist / np.linalg.norm(clip_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (clip_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()

            for i in range(self.patch_query_obj.shape[0]-1):
                binary_foreground = (patch_merge_sam == i).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
                for i in range(1, num_labels):
                    instance_mask = (labels == i).astype(np.uint8)
                    instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                    if instance_mask.any():
                        instance_masks.append(instance_mask.astype(np.bool).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

                # plt.axis('off')
                # plt.imshow(patch_merge_sam)

                # plt.savefig('pic/vis/{}_seg_{}.png'.format(class_name, path), bbox_inches='tight', pad_inches = 0) # pad_inches = 0
                # plt.close()
                
            
            return {"score": score, "foreground_pixel_count": foreground_pixel_count, "clip_patch_hist": clip_patch_hist, "instance_masks": instance_masks}

        elif self.class_name == 'breakfast_box':
            # patch hist
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0]) 
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T)
                score = 1 - patch_hist_similarity.max()
            
            # todo: exist of foreground

            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1)).astype(np.uint8)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))

            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]
            
            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show()

                # plt.axis('off')
                # plt.imshow(patch_merge_sam)

                # plt.savefig('pic/vis/{}_seg_{}.png'.format(class_name, path), bbox_inches='tight', pad_inches = 0) # pad_inches = 0
                # plt.close()

            return {"score": score, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}
        
        elif self.class_name == 'juice_bottle': 
            # remove noise due to non sam mask
            merge_sam[sam_mask == 0] = self.classes - 1
            patch_merge_sam[sam_mask == 0] = self.patch_query_obj.shape[0] - 1  # 79.5

            # [['glass'], ['liquid in bottle'], ['fruit'], ['label', 'tag'], ['black background', 'background']], 
            # fruit and liquid mismatch (todo if exist)
            resized_patch_merge_sam = cv2.resize(patch_merge_sam, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
            binary_liquid = (resized_patch_merge_sam == 1)
            binary_fruit = (resized_patch_merge_sam == 2)

            query_liquid = encode_obj_text(self.model_clip, self.juice_bottle_liquid_query_words_dict, self.tokenizer, self.device)
            query_fruit = encode_obj_text(self.model_clip, self.juice_bottle_fruit_query_words_dict, self.tokenizer, self.device)

            liquid_feature = proj_patch_token[binary_liquid.reshape(-1), :].mean(0, keepdim=True)
            liquid_idx = (liquid_feature @ query_liquid.T).argmax(-1).squeeze(0).item()

            fruit_feature = proj_patch_token[binary_fruit.reshape(-1), :].mean(0, keepdim=True)
            fruit_idx = (fruit_feature @ query_fruit.T).argmax(-1).squeeze(0).item()
            
            if (liquid_idx != fruit_idx) and self.anomaly_flag is False:
                print('liquid: {}, but fruit: {}.'.format(self.juice_bottle_liquid_query_words_dict[liquid_idx], self.juice_bottle_fruit_query_words_dict[fruit_idx]))
                self.anomaly_flag = True

            # # todo centroid of fruit and tag_0 mismatch (if exist) ,  only one tag, center

            # patch hist 
            sam_patch_hist = np.bincount(patch_merge_sam.reshape(-1), minlength=self.patch_query_obj.shape[0])
            sam_patch_hist = sam_patch_hist / np.linalg.norm(sam_patch_hist)

            if self.few_shot_inited:  
                patch_hist_similarity = (sam_patch_hist @ self.patch_token_hist.T) 
                score = 1 - patch_hist_similarity.max()
            
            binary_foreground = (patch_merge_sam != (self.patch_query_obj.shape[0] - 1) ).astype(np.uint8) 
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_foreground, connectivity=8)
            for i in range(1, num_labels):
                instance_mask = (labels == i).astype(np.uint8)
                instance_mask = cv2.resize(instance_mask, (self.feat_size, self.feat_size), interpolation = cv2.INTER_NEAREST)
                if instance_mask.any():
                    instance_masks.append(instance_mask.astype(np.bool).reshape(-1))
            
            if len(instance_masks) != 0:
                instance_masks = np.stack(instance_masks) #[N, 64x64]

            if self.visualization:
                image_list = [raw_image, kmeans_label, kmeans_mask, patch_mask, sam_mask, merge_sam, patch_merge_sam, binary_foreground]
                title_list = ['raw image', 'k-means', 'kmeans mask', 'patch mask', 'sam mask', 'merge sam mask', 'patch merge sam', 'binary_foreground']
                for ind, (temp_title, temp_image) in enumerate(zip(title_list, image_list), start=1):
                    plt.subplot(1, len(image_list), ind)
                    plt.imshow(temp_image)
                    plt.title(temp_title)
                    plt.margins(0, 0)
                    plt.axis('off')
                plt.show() 

            return {"score": score, "sam_patch_hist": sam_patch_hist, "instance_masks": instance_masks}

        return {"score": score, "instance_masks": instance_masks}


    def process_k_shot(self, class_name, few_shot_samples, few_shot_paths):
        few_shot_samples = F.interpolate(few_shot_samples, size=(448, 448), mode=self.inter_mode, align_corners=self.align_corners, antialias=self.antialias)

        with torch.no_grad():
            image_features, patch_tokens, proj_patch_tokens = self.model_clip.encode_image(few_shot_samples, self.feature_list)
            patch_tokens = [p[:, 1:, :] for p in patch_tokens]
            patch_tokens = [p.reshape(p.shape[0]*p.shape[1], p.shape[2]) for p in patch_tokens]

            patch_tokens_clip = torch.cat(patch_tokens, dim=-1)  # (bs, 1024, 1024x4)
            # patch_tokens_clip = torch.cat(patch_tokens[2:], dim=-1)  # (bs, 1024, 1024x2)
            patch_tokens_clip = patch_tokens_clip.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_clip = F.interpolate(patch_tokens_clip, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_clip = patch_tokens_clip.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.feature_list))
            patch_tokens_clip = F.normalize(patch_tokens_clip, p=2, dim=-1)  # (bsx64x64, 1024x4)

        with torch.no_grad():
            patch_tokens_dinov2 = self.model_dinov2.forward_features(few_shot_samples, out_layer_list=self.feature_list_dinov2)  # 4 x [bs, 32x32, 1024]
            patch_tokens_dinov2 = torch.cat(patch_tokens_dinov2, dim=-1)  # (bs, 1024, 1024x4)
            patch_tokens_dinov2 = patch_tokens_dinov2.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            patch_tokens_dinov2 = F.interpolate(patch_tokens_dinov2, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            patch_tokens_dinov2 = patch_tokens_dinov2.permute(0, 2, 3, 1).view(-1, self.vision_width_dinov2 * len(self.feature_list_dinov2))
            patch_tokens_dinov2 = F.normalize(patch_tokens_dinov2, p=2, dim=-1)  # (bsx64x64, 1024x4)

        
        cluster_features = None
        for layer in self.cluster_feature_id:
            temp_feat = patch_tokens[layer]
            cluster_features = temp_feat if cluster_features is None else torch.cat((cluster_features, temp_feat), 1)
        if self.feat_size != self.ori_feat_size:
            cluster_features = cluster_features.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            cluster_features = F.interpolate(cluster_features, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            cluster_features = cluster_features.permute(0, 2, 3, 1).view(-1, self.vision_width * len(self.cluster_feature_id))
        cluster_features = F.normalize(cluster_features, p=2, dim=-1)

        if self.feat_size != self.ori_feat_size:
            proj_patch_tokens = proj_patch_tokens.view(self.k_shot, self.ori_feat_size, self.ori_feat_size, -1).permute(0, 3, 1, 2)
            proj_patch_tokens = F.interpolate(proj_patch_tokens, size=(self.feat_size, self.feat_size), mode=self.inter_mode, align_corners=self.align_corners)
            proj_patch_tokens = proj_patch_tokens.permute(0, 2, 3, 1).view(-1, self.embed_dim)
        proj_patch_tokens = F.normalize(proj_patch_tokens, p=2, dim=-1)

        if not self.cluster_init:
            num_clusters = self.cluster_num_dict[class_name]
            _, self.cluster_centers = kmeans(X=cluster_features, num_clusters=num_clusters, device=self.device)
    
            self.query_obj = encode_obj_text(self.model_clip, self.query_words_dict[class_name], self.tokenizer, self.device)
            self.patch_query_obj = encode_obj_text(self.model_clip, self.patch_query_words_dict[class_name], self.tokenizer, self.device)
            self.classes = self.query_obj.shape[0]

            self.cluster_init = True

        scores = []
        foreground_pixel_hist = []
        splicing_connectors_distance = []
        patch_token_hist = []
        mem_instance_masks = []
            
        for image, cluster_feature, proj_patch_token, few_shot_path in zip(few_shot_samples.chunk(self.k_shot), cluster_features.chunk(self.k_shot), proj_patch_tokens.chunk(self.k_shot), few_shot_paths):        
            # path = os.path.dirname(few_shot_path).split('/')[-1] + "_" + os.path.basename(few_shot_path).split('.')[0]
            self.anomaly_flag = False
            results = self.histogram(image, cluster_feature, proj_patch_token, class_name, "few_shot_" + os.path.basename(few_shot_path).split('.')[0])
            if self.class_name == 'pushpins':
                patch_token_hist.append(results["clip_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'splicing_connectors':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                splicing_connectors_distance.append(results["distance"])
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'screw_bag':
                foreground_pixel_hist.append(results["foreground_pixel_count"])
                patch_token_hist.append(results["clip_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'breakfast_box':
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            elif self.class_name == 'juice_bottle':
                patch_token_hist.append(results["sam_patch_hist"])
                mem_instance_masks.append(results['instance_masks'])

            scores.append(results["score"])

        if len(foreground_pixel_hist) != 0:
            self.foreground_pixel_hist = np.mean(foreground_pixel_hist)
        if len(splicing_connectors_distance) != 0:
            self.splicing_connectors_distance = np.mean(splicing_connectors_distance)
        if len(patch_token_hist) != 0: # patch hist
            self.patch_token_hist = np.stack(patch_token_hist)
        if len(mem_instance_masks) != 0:
            self.mem_instance_masks = mem_instance_masks

        # for interests matching
        len_feature_list = len(self.feature_list)
        for idx, batch_mem_patch_feature in enumerate(patch_tokens_clip.chunk(len_feature_list, dim=-1)): # 4 stages batch_mem_patch_feature (bsx64x64, 1024)
            mem_instance_features = []
            for mem_patch_feature, mem_instance_masks in zip(batch_mem_patch_feature.chunk(self.k_shot), self.mem_instance_masks): # k shot  mem_patch_feature (64x64, 1024)
                mem_instance_features.extend([mem_patch_feature[mask, :].mean(0, keepdim=True) for mask in mem_instance_masks])
            mem_instance_features = torch.cat(mem_instance_features, dim=0)
            mem_instance_features = F.normalize(mem_instance_features, dim=-1) # 4 stages
            # mem_instance_features_multi_stage.append(mem_instance_features)
            self.mem_instance_features_multi_stage[idx].append(mem_instance_features)


        mem_patch_feature_clip_coreset = patch_tokens_clip
        mem_patch_feature_dinov2_coreset = patch_tokens_dinov2

        return scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset
    
    def process(self, class_name: str, few_shot_samples: list[torch.Tensor], few_shot_paths: list[str]):
        few_shot_samples = self.transform(few_shot_samples).to(self.device)

        scores, mem_patch_feature_clip_coreset, mem_patch_feature_dinov2_coreset = self.process_k_shot(class_name, few_shot_samples, few_shot_paths)

        clip_sampler = KCenterGreedy(embedding=mem_patch_feature_clip_coreset, sampling_ratio=0.25)
        mem_patch_feature_clip_coreset = clip_sampler.sample_coreset()

        dinov2_sampler = KCenterGreedy(embedding=mem_patch_feature_dinov2_coreset, sampling_ratio=0.25)
        mem_patch_feature_dinov2_coreset = dinov2_sampler.sample_coreset()

        self.mem_patch_feature_clip_coreset.append(mem_patch_feature_clip_coreset)
        self.mem_patch_feature_dinov2_coreset.append(mem_patch_feature_dinov2_coreset)


    def setup(self, data: dict) -> None:
        """Setup the few-shot samples for the model.

        The evaluation script will call this method to pass the k images for few shot learning and the object class
        name. In the case of MVTec LOCO this will be the dataset category name (e.g. breakfast_box). Please contact
        the organizing committee if if your model requires any additional dataset-related information at setup-time.
        """
        few_shot_samples = data.get("few_shot_samples")
        class_name = data.get("dataset_category")
        few_shot_paths = data.get("few_shot_samples_path")
        self.class_name = class_name

        print(few_shot_samples.shape)

        self.total_size = few_shot_samples.size(0)

        self.k_shot = 4 if self.total_size > 4 else self.total_size

        self.cluster_init = False
        self.mem_instance_features_multi_stage = [[],[],[],[]]

        self.mem_patch_feature_clip_coreset = []
        self.mem_patch_feature_dinov2_coreset = []

        if self.save_coreset_features:
            for i in range(self.total_size//self.k_shot):
                self.process(class_name, few_shot_samples[self.k_shot*i : min(self.k_shot*(i+1), self.total_size)], few_shot_paths[self.k_shot*i : min(self.k_shot*(i+1), self.total_size)])

            # Coreset Subsampling
            self.mem_patch_feature_clip_coreset = torch.cat(self.mem_patch_feature_clip_coreset, dim=0)
            torch.save(self.mem_patch_feature_clip_coreset, 'memory_bank/mem_patch_feature_clip_{}.pt'.format(self.class_name))

            self.mem_patch_feature_dinov2_coreset = torch.cat(self.mem_patch_feature_dinov2_coreset, dim=0)
            torch.save(self.mem_patch_feature_dinov2_coreset, 'memory_bank/mem_patch_feature_dinov2_{}.pt'.format(self.class_name))

            print(self.mem_patch_feature_dinov2_coreset.shape, self.mem_patch_feature_clip_coreset.shape)

            self.mem_instance_features_multi_stage = [ torch.cat(mem_instance_features, dim=0) for mem_instance_features in self.mem_instance_features_multi_stage ]
            self.mem_instance_features_multi_stage = torch.cat(self.mem_instance_features_multi_stage, dim=1)
            torch.save(self.mem_instance_features_multi_stage, 'memory_bank/mem_instance_features_multi_stage_{}.pt'.format(self.class_name))
            
            print(self.mem_instance_features_multi_stage.shape)

        else:
            self.process(class_name, few_shot_samples[0 : self.k_shot], few_shot_paths[0 : self.k_shot])

            self.mem_patch_feature_clip_coreset = torch.load('memory_bank/mem_patch_feature_clip_{}.pt'.format(self.class_name))
            self.mem_patch_feature_dinov2_coreset = torch.load('memory_bank/mem_patch_feature_dinov2_{}.pt'.format(self.class_name))
            self.mem_instance_features_multi_stage = torch.load('memory_bank/mem_instance_features_multi_stage_{}.pt'.format(self.class_name))


        self.few_shot_inited = True
        

