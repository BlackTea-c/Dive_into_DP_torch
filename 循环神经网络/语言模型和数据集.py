


'''
在自然语言处理（NLP）中，一元语法（Unigram）、二元语法（Bigram）、和 n 元语法（N-gram）是用于建模文本或语言的不同级别的技术。
一元语法（Unigram）：一元语法是指将文本中的每个单词或标记视为一个独立的单元，不考虑其前后文本的关系。
一元语法假设每个单词都是独立的，因此它忽略了单词之间的上下文关系。例如，在情感分析中，对每个单词进行独立分类而不考虑它与周围单词的联系就是一元语法的一种形式。
二元语法（Bigram）：二元语法考虑相邻的两个单词作为一个单元来建模语言。它假设一个单词出现的概率依赖于其前面出现的一个单词。
通过考虑相邻单词的组合，二元语法能够捕捉到一些局部的语言结构和上下文信息。例如，在语言建模中，计算句子中每个单词出现的概率时，考虑前一个单词的信息就是二元语法的应用。
n 元语法（N-gram）：n 元语法是一种更一般化的方法，其中 n 表示考虑的单位数量。除了一元和二元语法外，还可以有三元语法（Trigram，考虑相邻的三个单词）、
四元语法（Quadgram，考虑相邻的四个单词）等等。n 元语法尝试通过考虑更长的上下文序列来更好地建模语言的结构和语境，以提高模型的性能和准确性。

这些语法模型通常用于语言建模、文本生成、自动文本分类、机器翻译等 NLP 任务中，通过对文本数据中的不同级别的语言单元进行建模，以提取并理解语言中的结构和含义。'''


import 文本预处理 as d2l


import random
import torch


tokens = d2l.tokenize(d2l.read_time_machine())
# 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
print('一元:',vocab.token_freqs[:10]) #打印出频率最高的几个单词 此为一元语法下的

'''正如我们所看到的，最流行的词看起来很无聊， 这些词通常被称为停用词（stop words），因此可以被过滤掉。 尽管如此，它们本身仍然是有意义的，
我们仍然会在模型中使用它们。 
此外，还有个明显的问题是词频衰减的速度相当地快。 例如，最常用单词的词频对比，第
个还不到第10个的0.2
。 为了更好地理解，我们可以画出的词频图：'''

#现在分别看看二元，三元：
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
print('二元',bigram_vocab.token_freqs[:10])

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
print('三元：',trigram_vocab.token_freqs[:10])

#corpus就是文本（单词转成了索引） 参数num_steps是每个子序列中预定义的时间步数。
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


#顺序分区法
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y



class SeqDataLoader:  #@save
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')