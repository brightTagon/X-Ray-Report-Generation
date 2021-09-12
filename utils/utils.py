import torch
import numpy as np
import pandas as pd
import cv2
import os
import re
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def show_img(img, vmin, vmax, cmap='gray'):
    plt.figure(figsize=(16,10))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis(False)
    plt.show()

def transform_img(img, transform):
    img = transform(img).unsqueeze(0)
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
    return img

def weight_equal_by_instance(lengths):
    """
    weight such that each instance in a batch has an equal weight
    Return: weight (n,)
    """
    weight = 1 / lengths
    weight = weight[lengths_to_idx(lengths)]
    weight = weight / weight.sum()
    return weight


def mean_equal_by_instance(x, lengths):
    """
    Args:
        x: (n, hid)
        lengths: (bs, )
    """
    assert x.dim() == 2
    weight = weight_equal_by_instance(lengths)
    _, hid = x.shape
    x = (x * weight.unsqueeze(-1) / hid).sum()
    return x


def lengths_to_idx(lengths):
    """
    [1, 2] into [0, 1, 1]
    """
    idx = []
    for i, length in enumerate(lengths):
        idx += [i] * length
    return torch.LongTensor(idx).to(lengths.device)

def cut_startseq_endseq(sentence):
    return sentence[8:-6]

def strip_sentence(sentence, char=None):
    return sentence.strip(char)

def split_sentences(sentences):
    sentences = cut_startseq_endseq(sentences)
    _, clean_sentences = clean_report(sentences)
#     list_sentence = clean_sentences.split(' . ')
#     if(len(list_sentence) > 1 and list_sentence[-1] == ''):
#         list_sentence.pop()
    return list(map(strip_sentence, clean_sentences))

def clean_report(report):
    report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
        .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
        .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub(
        '[.,?;*!%^&_+():-\[\]{}]', '',
        t.replace('"', '').replace('/', '').replace('\\', '').replace(
            "'", '').strip().lower())
    tokens = [
        sent_cleaner(sent) for sent in report_cleaner(report)
        if sent_cleaner(sent) != []
    ]
    report = ' . '.join(tokens) + ' . '
    return report, tokens
    