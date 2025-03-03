import os
from typing import Union, List
import torch
import numpy as np
from tqdm import tqdm
from imagenet_template import openai_imagenet_template


def encode_text_with_prompt_ensemble(model, objs, tokenizer, device):
    prompt_normal = ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', '{} without flaw', '{} without defect', '{} without damage']
    prompt_abnormal = ['damaged {}', 'broken {}', '{} with flaw', '{} with defect', '{} with damage']
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = ['a bad photo of a {}.', 'a low resolution photo of the {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a bright photo of a {}.', 'a dark photo of the {}.', 'a photo of my {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a photo of one {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'a low resolution photo of a {}.', 'a photo of a large {}.', 'a blurry photo of a {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a photo of the small {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'a dark photo of a {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']
    text_prompts = {}
    for obj in objs:
        text_features = []
        for i in range(len(prompt_state)):
            prompted_state = [state.format(obj) for state in prompt_state[i]]
            prompted_sentence = []
            for s in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(s))
            prompted_sentence = tokenizer(prompted_sentence).to(device)
            class_embeddings = model.encode_text(prompted_sentence)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            text_features.append(class_embedding)

        text_features = torch.stack(text_features, dim=1).to(device)
        text_prompts[obj] = text_features

    return text_prompts


def encode_general_text(model, obj_list, tokenizer, device):
    text_dir = '/data/yizhou/VAND2.0/wgd/general_texts/train2014'
    text_name_list = sorted(os.listdir(text_dir))
    bs = 100
    sentences = []
    embeddings = []
    all_sentences = []
    for text_name in tqdm(text_name_list):
        with open(os.path.join(text_dir, text_name), 'r') as f:
            for line in f.readlines():
                sentences.append(line.strip())
        if len(sentences) > bs:
            prompted_sentences = tokenizer(sentences).to(device)
            class_embeddings = model.encode_text(prompted_sentences)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            embeddings.append(class_embeddings)
            all_sentences.extend(sentences)
            sentences = []
        # if len(all_sentences) > 10000:
        #     break
    embeddings = torch.cat(embeddings, 0)
    print(embeddings.size(0))
    embeddings_dict = {}
    for obj in obj_list:
        embeddings_dict[obj] = embeddings
    return embeddings_dict, all_sentences


def encode_abnormal_text(model, obj_list, tokenizer, device):
    embeddings = {}
    sentences = {}
    for obj in obj_list:
        sentence_abnormal = []
        with open(os.path.join('text_prompt', 'v1', obj + '_abnormal.txt'), 'r') as f:
            for line in f.readlines():
                sentence_abnormal.append(line.strip().lower())

        prompted_sentences = tokenizer(sentence_abnormal).to(device)
        class_embeddings = model.encode_text(prompted_sentences)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        embeddings[obj] = class_embeddings
        sentences[obj] = sentence_abnormal
    return embeddings, sentences


def encode_normal_text(model, obj_list, tokenizer, device):
    embeddings = {}
    sentences = {}
    for obj in obj_list:
        sentence_abnormal = []
        with open(os.path.join('text_prompt', 'v1', obj + '_normal.txt'), 'r') as f:
            for line in f.readlines():
                sentence_abnormal.append(line.strip().lower())

        prompted_sentences = tokenizer(sentence_abnormal).to(device)
        class_embeddings = model.encode_text(prompted_sentences)
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        embeddings[obj] = class_embeddings
        sentences[obj] = sentence_abnormal
    return embeddings, sentences


def encode_obj_text(model, query_words, tokenizer, device):
    # query_words = ['orange', "nectarine", "cereals", "banana chips", 'almonds', 'white box']
    # query_words = ['liquid', 'glass', "top", 'black background']
    # query_words = ["connector", "grid"]
    # query_words = [['screw'], 'plastic bag', 'background']
    # query_words = [['pushpin', 'pin'], ['plastic box'], 'box', 'black background']
    query_features = []
    with torch.no_grad():
        for qw in query_words:
            token_input = []
            if type(qw) == list:
                for qw2 in qw:
                    token_input.extend([temp(qw2) for temp in openai_imagenet_template])
            else:
                token_input = [temp(qw) for temp in openai_imagenet_template]
            query = tokenizer(token_input).to(device)
            feature = model.encode_text(query)
            feature /= feature.norm(dim=-1, keepdim=True)
            feature = feature.mean(dim=0)
            feature /= feature.norm()
            query_features.append(feature.unsqueeze(0))
    query_features = torch.cat(query_features, dim=0)
    return query_features

