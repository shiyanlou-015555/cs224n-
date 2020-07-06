

from tqdm import tqdm
import numpy as np
from transformers import AdamW

accumulate_gradients=2

class DataGenerator(object):
    def __init__(self, x_data,y_data,tokenizer, max_seq_length,batch_size, device):
        perm = np.random.permutation(len(y_data))#random shuffle随机打乱
        self.x_data = x_data[perm]
        self.y_data = y_data[perm]
        self.tokenizer = tokenizer
        self.max_seq_length=max_seq_length
        self.batch_size = batch_size
        self.device = device

    def __next__(self):
        pass
    def __iter__(self):
        return self

    def __len__(self):
        pass

    def generator(self,data):
        idxs = list(range(len(data)))
        input_ids = []
        segment_ids = []
        input_mask = []
        for text in idxs:
            tokens_a=self.tokenizer.tokenize(text)
            if len(tokens_a) > self.max_seq_length - 2:
                tokens_a = tokens_a[:(self.max_seq_length - 2)]
            tokens1 = ['[CLS]'] + tokens_a + ['[SEP]']
            segment_ids1 = [0] * len(tokens1)
            input_ids1 = self.tokenizer.convert_tokens_to_ids(tokens1)
            input_mask1 = [1] * len(input_ids1)

            # input_ids：一个形状为[batch_size, sequence_length]的torch.LongTensor，在词汇表中包含单词的token索引
            # segment_ids ：形状[batch_size, sequence_length]的可选torch.LongTensor，在[0, 1]中选择token类型索引。类型0对应于句子A，类型1对应于句子B。
            # input_mask：一个可选的torch.LongTensor，形状为[batch_size, sequence_length]，索引在[0, 1]中选择。

            padding = [0] * (self.max_seq_length - len(input_ids1))
            input_ids1 += padding
            input_mask1 += padding
            segment_ids1 += padding
            input_ids.append(input_ids1)  # bert的输入
            segment_ids.append(segment_ids1)  # bert的输入
            input_mask.append(input_mask1)# bert的输入
            # Zero-pad up to the sequence length.
        return input_ids, input_mask, segment_ids

def train(model,train_loader,tokenizer,learning_rate,num_epocs,device):
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    opt = AdamW(optimizer_grouped_parameters,
                         lr=learning_rate
                )

    for epch in tqdm(range(int(num_epocs)), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for index, t in enumerate(tqdm(train_loader, desc="Iteration")):

            input_ids, input_mask, segment_ids, label_ids = generator(t,y,tokenizer)
            loss = model(input_ids, segment_ids, input_mask, label_ids)

            # Calculate gradient
            if index % accumulate_gradients == 0:  # mode
                # at start, perform zero_grad
                opt.zero_grad()
                loss.backward()
                if accumulate_gradients == 1:
                    opt.step()
            elif index % accumulate_gradients == (accumulate_gradients - 1):
                # at the final, take step with accumulated graident
                loss.backward()
                opt.step()
            else:
                # at intermediate stage, just accumulates the gradients
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1


        print('Loss after epoc {}'.format(tr_loss / nb_tr_steps))