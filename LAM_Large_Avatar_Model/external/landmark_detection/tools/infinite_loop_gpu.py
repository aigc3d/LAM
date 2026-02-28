# -*- coding: utf-8 -*-

import os
import time
import torch
import argparse

parser = argparse.ArgumentParser(description='inf')
parser.add_argument('--gpu', default='1', type=str, help='index of gpu to use')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

n = 1000

x = torch.zeros(4, n, n).cuda()
rest_time = 0.0000000000001
while True:
    y = x * x
    time.sleep(rest_time)
    y1 = x * x
