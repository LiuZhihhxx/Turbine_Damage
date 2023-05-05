import argparse
import torch
import numpy as np


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='t4_2.csv', help='type of optimizer')  # wtbdata_245days.csv
    parser.add_argument('--number_classes', type=str, default='t4_2.csv', help='type of optimizer')
    # [3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # 输入输出长度/维度声明
    parser.add_argument('--feature_col', type=int, default=np.linspace(3, 15, 15 - 2, dtype=int), help='feature_col')
    parser.add_argument('--input_dim', type=int, default=len(parser.parse_args().feature_col) + 1,
                        help='input dimension')

    # parser.add_argument('--start_col', type=int, default=3, help='start_col')1

    # parser.add_argument('--end_col', type=int, default=12, help='end_col')
    # parser.add_argument('--input_dim', type=int, default=parser.parse_args().end_col - parser.parse_args().start_col + 1, help='input dimension')

    parser.add_argument('--input_len', type=int, default=24 * 6, help='input length')
    parser.add_argument('--output_len', type=int, default=4 * 6, help='output length')

    # 网络超参数声明
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='num layers')

    # 优化器超参数声明
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')  # 正则化系数
    parser.add_argument('--bidirectional', type=bool, default=False, help='LSTM direction')  # 双向RNN
    parser.add_argument('--step_size', type=int, default=10, help='step size')  # 每进行step_size次训练，学习率调整一次
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma')  # 学习率调整权重
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()

    return args
