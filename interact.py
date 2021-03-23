import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
import logging
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
from os.path import join, exists
from itertools import zip_longest, chain
# from chatbot.model import DialogueGPT2Model
from dataset import MyDataset
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from train import create_model
import torch.nn.functional as F

PAD = '[PAD]'
pad_id = 0


def set_interact_args():
    """
    Sets up the training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成的temperature')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高k选1')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--model_config', default='chinese-lyric-gpt-pretrain-model/config.txt', type=str,
                        required=False,
                        help='模型参数')
    parser.add_argument('--log_path', default='data/interacting.log', type=str, required=False, help='interact日志存放位置')
    parser.add_argument('--voca_path', default='chinese-lyric-gpt-pretrain-model/vocab.txt', type=str, required=False,
                        help='选择词库')
    parser.add_argument('--model_path', default='chinese-lyric-gpt-pretrain-model/', type=str, required=False,
                        help='模型路径')
    parser.add_argument('--save_samples_path', default="sample/", type=str, required=False, help="保存记录的文件路径")
    parser.add_argument('--load_form_path', default="sample/form.txt", type=str, required=False, help="读取期望输入歌词格式的文件路径")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False,
                        help="重复惩罚参数，若生成的对话重复性较高，可适当提高该参数")
    parser.add_argument('--seed', type=int, default=None, help='设置种子用于生成随机数，以使得训练的结果是确定的')
    parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
    parser.add_argument('--max_history_len', type=int, default=3, help="生成句子 history的最大长度")
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    return parser.parse_args()


def create_logger(args):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(
        filename=args.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    args = set_interact_args()
    logger = create_logger(args)
    # 当用户使用GPU,并且GPU可用时
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    device = 'cuda' if args.cuda else 'cpu'
    logger.info('using device:{}'.format(device))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    tokenizer = BertTokenizer(vocab_file=args.voca_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/samples.txt', 'a', encoding='utf8')
        samples_file.write("记录{}:\n".format(datetime.now()))
        # 存储记录，每个utterance以token的id的形式进行存储

    # 读取期望生成的歌词格式
    with open(args.load_form_path, "r", encoding="utf8") as f:
        data = f.read()
    data_list = data.split("\n")
    song_name = data_list[0].replace('歌名：', '')
    lyrics = data_list[1:]
    samples_file.write("歌名:{}\n".format(song_name))

    history = []

    for lyric_ids, lyric in enumerate(lyrics):

        # 如果下一句是非空，则本句话不用生成下一句，直接跳过
        if '[]' not in lyrics[lyric_ids + 1]:
            continue
        else:
            # 获得下一句要生成多少个字
            next_sent_len = len(lyrics[lyric_ids + 1]) / 2

        # 如果本句不需要生成，就用已有句子代替
        if '[]' not in lyric:
            text = lyric
            if args.save_samples_path:
                samples_file.write("input:({})\n".format(text))
        # 记住生成歌词的最大长度，如果大于最大长度则删除第一句然后再添加 history :[set1,set2,set3,set4]
        if len(history) > args.max_history_len:
            history.pop(0)
            history.append([tokenizer.encode(text)])
        else:
            history.append([tokenizer.encode(text)])

        # 每个input以  [CLS]歌名[sep]为开头
        input_ids = [tokenizer.cls_token_id]
        input_ids.extend(tokenizer.encode(song_name))
        input_ids.append(tokenizer.sep_token_id)

        for his_setences in history:
            for setence in his_setences:
                input_ids.extend(setence)
                input_ids.append(tokenizer.sep_token_id)
        curr_input_tensor = torch.tensor(input_ids).long().to(device)
        generated = []
        token_ids = 1  # 生成的第 token_ids 个字
        generated_backup = {}
        # 最多生成next_sent_len+1个token
        while token_ids <= next_sent_len + 1:
            outputs = model(input_ids=curr_input_tensor)
            next_token_logits = outputs[0][-1, :]
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id in set(generated):
                next_token_logits[id] /= args.repetition_penalty
            next_token_logits = next_token_logits / args.temperature
            # 对于[UNK]的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
            # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
            next_tokens = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=3)
            generated_backup[token_ids+1] = next_tokens.copy()
            if  next_sent_len - (token_ids + 1)   <= 0  :#如果生成的位置处于容许范围内，且没有sep
                pass
                # 删除当前backup的【0】元素，重新生成next token
            for next_token in next_tokens:
                next_token = next_token.unsqueeze(0)
                if next_token == tokenizer.sep_token_id or next_token == torch.tensor(tokenizer.encode('，')).to(
                        device):  # 遇到[SEP]或者逗号则表明response生成结束
                    break
                generated.append(next_token.item())
                curr_input_tensor = torch.cat((curr_input_tensor, next_token), dim=0)
            # his_text = tokenizer.convert_ids_to_tokens(curr_input_tensor.tolist())
            # print("his_text:{}".format(his_text))
        # history.append(generated)
        text = tokenizer.convert_ids_to_tokens(generated)
        print("bot:" + "".join(text))
        if args.save_samples_path:
            samples_file.write("bot:{}\n".format("".join(text)))


if __name__ == '__main__':
    main()
