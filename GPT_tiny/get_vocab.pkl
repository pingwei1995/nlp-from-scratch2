# 该文件用于生成字典信息,读取dataset.txt文件(由generate.py生成)，生成dict_datas.json文件,这个文件就有两个字典信息，也就是新字典(word2id)和列表(id2word)

import json

def get_dict(datas): #接收dataset.txt文件的信息，返回字典(word2id)和列表(id2word)，并打印字频
    word_count ={} #字典，用于再第二层循环中统计字频，作为字键对应的值
    for data in datas: #遍历dataset.txt的每一行数据
        data = data.strip().replace('\t','')
        for word in data: #遍历每一个字
            word_count.setdefault(word,0) #如果当前字典没有这个键，则添加这个键，默认值0。如果有这个键，不发挥作用
            word_count[word]+=1  #统计字频
    word2id = {"<pad>":0,"<unk>":1,"<sep>":2} #新字典，下面会为每个字单独分配一个独一无二的数字
    temp = {word: i + len(word2id) for i, word in enumerate(word_count.keys())}
    word2id.update(temp) #字典update方法并入新字典。{字:数字...}
    id2word=list(word2id.keys()) #列表，索引取出的值就是对应的字

    # 打印字频情况
    high = sorted(word_count.items(),key=lambda x:x[1],reverse=True)[:10] #用词频来降序排序
    low  = sorted(word_count.items(),key=lambda x:x[1],reverse=False)[:10]
    print('最常出现的10个字：',high)
    print('最不常出现的10个字：',low)

    return word2id,id2word #返回新字典(word2id)和列表(id2word)


if __name__ == '__main__':
    with open('dataset.txt','r',encoding='utf-8') as f:
        datas = f.readlines()
    word2id, id2word = get_dict(datas)

    # 把上面方法返回的字典和列表写入json文件(utf-8格式)
    dict_datas = {"word2id":word2id,"id2word":id2word}
    json.dump(dict_datas, open('dict_datas.json', 'w', encoding='utf-8'))
