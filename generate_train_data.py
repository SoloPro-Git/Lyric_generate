# coding:utf-8
import os
import re
from tqdm import tqdm
import json

def clear_lyric(name, context):
    name = name.split('.')[0]
    re_name = re.compile(r'\[(^())*\]')
    name = re.sub(re_name, '', name)
    name = name.replace('(Live)', '').replace('live', '').replace('-', '').replace('_', '').replace('(版)', '')
    lyric_cleared = '-------\n' + name + '\n'
    igonre_words = ['编曲', '作词', '作曲', '贝斯', '钢琴', '吉他', '键盘', '打击乐', '录音', '工程', '制作', '打击', 'by', '和声', '合声', '作 曲',
                    '作 词','鼓','歌曲','歌手','Bass','Piano','混音','Scratch','二胡','os','弦乐','Violin','Cello','Viola','violin']
    for i, line in enumerate(context):
        if i <= 15 and any(word in line for word in igonre_words):
            continue
        re_time = re.compile(r'\[[^()]*\]')
        line = re.sub(re_time,'',line)
        # line = line[10:]
        if line == '\n':
            continue
        lyric_cleared += line.replace(' ','\n')
    return lyric_cleared


def walkFile(file):
    lyric_list = []
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for filename in tqdm(files):
            with open(root + '/' + filename, 'r',encoding='utf-8') as f:
                context = f.readlines()
                lyric = clear_lyric(filename, context)
            lyric_list.append(lyric)
    return lyric_list

lyric_list = walkFile('data/lyric/周杰伦')
with open("data/train_Jay.json","w",encoding='utf-8') as f:
    for lyric_song in lyric_list:
        f.write(lyric_song)
    # json.dump(lyric_list,f)
