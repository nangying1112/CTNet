# -*- coding: utf-8 -*-

import time
import pickle
import numpy as np
import random
import os
# from utils import pp
import torch
from allennlp.commands.elmo import ElmoEmbedder

path = r'processed_data/'


class pddata():
    def __init__(self, FLAGS, relation=None):
        if os.path.exists(os.path.join(path, 'exp_imp_vocab_embed')) and os.path.exists(os.path.join(path, 'word2id')):
            self.embedding = pickle.load(open(os.path.join(path, 'exp_imp_vocab_embed'), 'rb'))
            self.voc2id = pickle.load(open(os.path.join(path, 'word2id'), 'rb'))
        self.train = None
        self.test = None
        self.use_elmo = FLAGS.use_elmo
        self.use_exp = FLAGS.use_exp
        self.elmo_cuda = FLAGS.elmo_cuda
        self.relation_balance = FLAGS.relation_balance
        self.type_balance = FLAGS.type_balance
        self.flag = FLAGS
        # elmo
        if self.use_elmo == True:
            options_file = path + 'options.json'
            weight_file = path + "weights.hdf5"
            self.elmo = ElmoEmbedder(options_file, weight_file,  cuda_device=FLAGS.elmo_cuda)
        self.class_layer1 = ['Expansion', 'Contingency', 'Comparison', 'Temporal']

    # for binary cls
    def gen_rel_data(self, relation):
        
        self.data = []
        exp_train = pickle.load(open(os.path.join(path, 'train_exp_para_adj'), 'rb'))
        imp_train = pickle.load(open(os.path.join(path, 'train_imp_para_adj'), 'rb'))
        train_dict = {}
        train_dict['exp'] = exp_train
        train_dict['imp'] = imp_train
        self.data.append(train_dict)
       
        devset = pickle.load(open(os.path.join(path, 'dev_para_adj'), 'rb'))
        self.data.append(devset)
        testset = pickle.load(open(os.path.join(path, 'test_para_adj'), 'rb'))
        self.data.append(testset)
        conn2id = pickle.load(open(os.path.join(path, 'conn2id'), 'rb'))
        
        type2id = {'Explicit': 0, 'Implicit': 1}

        train = {'imp':[], 'exp':[]}
        for sample in self.data[0]['imp']:
            # print(len(self.data[0]['imp']))
            for l in sample['label']:
                train['imp'].append([sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                  int(l==relation), sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()],sample['core_adj'], sample['word_adj'], sample['conn_adj'],  1])
        for sample in self.data[0]['exp']:
            # print(len(self.data[0]['imp']))
            for l in sample['label']:
                train['exp'].append([sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                  int(l==relation), sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()], sample['core_adj'], sample['word_adj'], sample['conn_adj'], 0])
      
        
        imp_neg = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], 0] for sample in train['imp'] if sample[6] == 0]
        imp_pos = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],1] for sample in train['imp'] if sample[6] == 1]
        # print('exp:', len(exp))
        exp_neg = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], 0] for sample in train['exp'] if
                   sample[6] == 0]
        exp_pos = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], 1] for sample in train['exp'] if
                   sample[6] == 1]

        self.tmp_data = {}
        if self.use_exp == True:
            self.tmp_data['train'] = [imp_pos, imp_neg, exp_pos, exp_neg]
        else:
            self.tmp_data['train'] = [imp_pos, imp_neg]
        print(len(self.data))
        # for sample in self.data[2]['imp']:
        #     print(sample)
        dev = [[sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1],
                 sample['pair'][0], sample['pair'][1],
                 sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()],
                 sample['core_adj'], sample['word_adj'], sample['conn_adj'], 1,
                 [int(rel == relation) for rel in sample['label']]] for sample in self.data[1]['imp']]
        self.tmp_data['dev'] = dev
        test = [[sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                 sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()], sample['core_adj'], sample['word_adj'], sample['conn_adj'], 1, 
                 [int(rel==relation) for rel in sample['label']]] for sample in self.data[2]['imp']]
        self.tmp_data['test'] = test 


    # for 4-way cls
    def gen_whole_data(self):

        # self.data = pickle.load(open(os.path.join(path, 'train_dev_test.ids'), 'rb'))
        #self.data = pickle.load(open(os.path.join(path, 'train_dev_test_para_slstm'), 'rb'))
        self.data = []
        exp_train = pickle.load(open(os.path.join(path, 'train_exp_para_adj'), 'rb'))
        imp_train = pickle.load(open(os.path.join(path, 'train_imp_para_adj'), 'rb'))
        train_dict = {}
        train_dict['exp'] = exp_train
        train_dict['imp'] = imp_train
        self.data.append(train_dict)

        devset = pickle.load(open(os.path.join(path, 'dev_para_adj'), 'rb'))
        self.data.append(devset)
        testset = pickle.load(open(os.path.join(path, 'test_para_adj'), 'rb'))
        self.data.append(testset)
        conn2id = pickle.load(open(os.path.join(path, 'conn2id'), 'rb'))

        rel2id = {'Expansion':0, 'Contingency':1, 'Comparison':2, 'Temporal':3}
        type2id = {'Explicit':0, 'Implicit':1}

        # train = [[sample['pair'][0], sample['pair'][1], rel2id[sample['label'][0]]] for sample in self.train_data]

        # arg1_ids, arg2_ids, arg1_char, arg2_char, arg1_words, arg2_words, label, para_ids, para_char,
        # para_words, conn_label, type_label
        train = {'imp':[], 'exp':[]}
        for sample in self.data[0]['imp']:
            # print(len(self.data[0]['imp']))
            for l in sample['label']:
                train['imp'].append([sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                  rel2id[l], sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()],sample['core_adj'], sample['word_adj'], sample['conn_adj'],  1])
        for sample in self.data[0]['exp']:
            # print(len(self.data[0]['imp']))
            for l in sample['label']:
                train['exp'].append([sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                  rel2id[l], sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()], sample['core_adj'], sample['word_adj'], sample['conn_adj'], 0])

        imp_exp = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],0] for sample in train['imp'] if sample[6] == 0]
        imp_con = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],1] for sample in train['imp'] if sample[6] == 1]
        imp_com = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11],sample[12], sample[13], sample[14], 2] for sample in train['imp'] if sample[6] == 2]
        imp_tem = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],3] for sample in train['imp'] if sample[6] == 3]
        # print('exp:', len(exp))
        exp_exp = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],0] for sample in train['exp'] if
                   sample[6] == 0]
        exp_con = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],1] for sample in train['exp'] if
                   sample[6] == 1]
        exp_com = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11], sample[12], sample[13], sample[14],2] for sample in train['exp'] if
                   sample[6] == 2]
        exp_tem = [[sample[0], sample[1], sample[2], sample[3], sample[4], sample[5],
                    sample[7], sample[8], sample[9], sample[10], sample[11],sample[12], sample[13], sample[14], 3] for sample in train['exp'] if
                   sample[6] == 3]


        self.tmp_data = {}
        if self.use_exp == True:
            self.tmp_data['train'] = [imp_exp, imp_con, imp_com, imp_tem, exp_exp, exp_con, exp_com, exp_tem]
        else:
            self.tmp_data['train'] = [imp_exp, imp_con, imp_com, imp_tem]
        print(len(self.data))
        dev = [[sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1],
                 sample['pair'][0], sample['pair'][1],
                 sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()],
                 sample['core_adj'], sample['word_adj'], sample['conn_adj'], 1,
                 [rel2id[rel] for rel in sample['label']]] for sample in self.data[1]['imp']]
        self.tmp_data['dev'] = dev
        test = [[sample['pair_ids'][0], sample['pair_ids'][1], sample['pair_char'][0], sample['pair_char'][1], sample['pair'][0], sample['pair'][1],
                 sample['para'], sample['para_ids'], sample['para_char'], conn2id[sample['conn'].lower()], sample['core_adj'], sample['word_adj'], sample['conn_adj'],1, [rel2id[rel] for rel in sample['label']]] for sample in self.data[2]['imp']]
        self.tmp_data['test'] = test

    # get a batch for binary cls
    def next_single_rel(self, batch_size, data_type='train', is_balance=False):

        if data_type == 'train':
            print(self.use_exp)
            if self.use_exp:
                if self.relation_balance == True and self.type_balance == True:
                    selected_samples = random.sample(self.tmp_data['train'][0], batch_size//4) + \
                                       random.sample(self.tmp_data['train'][1], batch_size//4) + \
                                       random.sample(self.tmp_data['train'][2], batch_size//4) + \
                                       random.sample(self.tmp_data['train'][3], batch_size//4) 


                elif self.relation_balance == False and self.type_balance == True:
                    imps = self.tmp_data['train'][0] + self.tmp_data['train'][1]
                    exps = self.tmp_data['train'][2] + self.tmp_data['train'][3]
                    selected_samples = random.sample(imps, batch_size // 2) + random.sample(exps, batch_size // 2)
                elif self.relation_balance == True and self.type_balance == False:
                    selected_samples = random.sample(self.tmp_data['train'][0]+self.tmp_data['train'][2], batch_size // 2) + \
                                      random.sample(self.tmp_data['train'][1]+self.tmp_data['train'][3], batch_size // 2) 

                else:
                    trains = self.tmp_data['train'][0] + self.tmp_data['train'][1] + self.tmp_data['train'][2] + \
                             self.tmp_data['train'][3]
                    selected_samples = random.sample(trains, batch_size)
            else:
                if self.relation_balance:
                    selected_samples = random.sample(self.tmp_data['train'][0], batch_size // 4) + \
                                       random.sample(self.tmp_data['train'][1], batch_size // 4) 
                else:
                    trains = self.tmp_data['train'][0] + self.tmp_data['train'][1]
                    selected_samples = random.sample(trains, batch_size)

            random.shuffle(selected_samples)
            return self._batch2input(selected_samples)

        elif data_type=='dev':
            selected_samples = self.tmp_data['dev']
            return selected_samples

        elif data_type=='test':
            selected_samples = self.tmp_data['test']
            return selected_samples

        else:
            return None

   # get a batch for 4-way cls
    def next_multi_rel(self, batch_size, data_type='train', is_balance=False):
        # is_balance: whether sampling in balance
        if data_type == 'train':
            print(self.use_exp)
            if self.use_exp:
                if self.relation_balance == True and self.type_balance == True:
                    selected_samples = random.sample(self.tmp_data['train'][0], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][1], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][2], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][3], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][4], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][5], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][6], batch_size//8) + \
                                       random.sample(self.tmp_data['train'][7], batch_size//8)


                elif self.relation_balance == False and self.type_balance == True:
                    imps = self.tmp_data['train'][0] + self.tmp_data['train'][1] + self.tmp_data['train'][2] + \
                             self.tmp_data['train'][3]
                    exps = self.tmp_data['train'][4] + self.tmp_data['train'][5] + self.tmp_data['train'][6] + \
                             self.tmp_data['train'][7]
                    selected_samples = random.sample(imps, batch_size // 2) + random.sample(exps, batch_size // 2)
                elif self.relation_balance == True and self.type_balance == False:
                    selected_samples = random.sample(self.tmp_data['train'][0]+self.tmp_data['train'][4], batch_size // 4) + \
                                      random.sample(self.tmp_data['train'][1]+self.tmp_data['train'][5], batch_size // 4) + \
                                      random.sample(self.tmp_data['train'][2]+self.tmp_data['train'][6], batch_size // 4) + \
                                      random.sample(self.tmp_data['train'][3]+self.tmp_data['train'][7], batch_size // 4)

                else:
                    trains = self.tmp_data['train'][0] + self.tmp_data['train'][1] + self.tmp_data['train'][2] + \
                             self.tmp_data['train'][3] +self.tmp_data['train'][4] + self.tmp_data['train'][5] + self.tmp_data['train'][6] + \
                             self.tmp_data['train'][7]
                    selected_samples = random.sample(trains, batch_size)
            else:
                if self.relation_balance:
                    selected_samples = random.sample(self.tmp_data['train'][0], batch_size // 4) + \
                                       random.sample(self.tmp_data['train'][1], batch_size // 4) + \
                                       random.sample(self.tmp_data['train'][2], batch_size // 4) + \
                                       random.sample(self.tmp_data['train'][3], batch_size // 4)
                else:
                    trains = self.tmp_data['train'][0] + self.tmp_data['train'][1] + self.tmp_data['train'][2] + \
                             self.tmp_data['train'][3]
                    selected_samples = random.sample(trains, batch_size)

            random.shuffle(selected_samples)
            return self._batch2input(selected_samples)
        elif data_type=='dev':
            selected_samples = self.tmp_data['dev']
            return selected_samples

        elif data_type=='test':
            selected_samples = self.tmp_data['test']
            return selected_samples
        else:
            return None


    def _padding(self, ids, max_len):
        # 'pad':0
        # print(ids)
        if len(ids) > max_len:
            # print(ids[-max_len:])
            return ids[-max_len:]
        else:
            # print((ids + [0]*max_len)[:max_len])
            return (ids + [0]*max_len)[:max_len]


    def _batch2input(self, selected_samples):

        # arg info
        arg_len = np.array(
            [[len(arg1), len(arg2)] for arg1, arg2, _, _, _, _, _, _, _, _, _, _, _, _, label in selected_samples])
        arg1_len = arg_len[:, 0]
        arg2_len = arg_len[:, 1]
        arg1_max_len = max(arg1_len)
        arg2_max_len = max(arg2_len)
        # arg1_max_len = 50
        # arg2_max_len = 50
        tmp = [[self._padding(arg1, arg1_max_len), self._padding(arg2, arg2_max_len), label]
               for arg1, arg2, _, _, _, _, _, _, _, _, _, _, _, _, label in selected_samples]
        arg1 = [arg1 for arg1, _, _ in tmp]
        arg2 = [arg2 for _, arg2, _ in tmp]
        arg1 = np.array(arg1)
        arg2 = np.array(arg2)
        # elmo
        if self.use_elmo == True:
            arg1_words = [sample[4].split() for sample in selected_samples]
            arg2_words = [sample[5].split() for sample in selected_samples]
            arg1_elmo, mask = self.elmo.batch_to_embeddings(arg1_words)
            arg2_elmo, mask = self.elmo.batch_to_embeddings(arg2_words)
            arg1_elmo = arg1_elmo.cpu()
            arg2_elmo = arg2_elmo.cpu()
        else:
            arg1_elmo = torch.ones([3])
            arg2_elmo = torch.ones([3])
        # char info
        char1 = []
        char2 = []
        for sample in selected_samples:
            arg1_char = sample[2]
            arg2_char = sample[3]
            while arg1_char.shape[0] != arg1_max_len:
                arg1_char = np.concatenate([arg1_char, np.zeros([1, 27])], 0)
            while arg2_char.shape[0] != arg2_max_len:
                arg2_char = np.concatenate([arg2_char, np.zeros([1, 27])], 0)
            char1.append(arg1_char)
            char2.append(arg2_char)
        char1 = np.array(char1)
        char2 = np.array(char2)

        supports = np.array([[sample[10], sample[11], sample[12]] for sample in selected_samples])

        label = [label for _, _, label in tmp]
        conn_label = [sample[-6] for sample in selected_samples]
        type_label = [sample[-2] for sample in selected_samples]

        # para info: para_ids, para_char, para_words (7, 8, 9)
        pad_para_chars, pad_para_ids, para_seq_len, pad_para_elmo = self._para2feature(selected_samples)

        return arg1, arg2, arg1_len, arg2_len, char1, char2, arg1_elmo, arg2_elmo, label, pad_para_chars, pad_para_ids, \
               pad_para_elmo, para_seq_len, conn_label, type_label, supports


    def _para2feature(self, selected_samples):

        # para2sentences
        all_sentences = []
        for sample in selected_samples:
            sentences = sample[6]
            # print(sample[7])
            # print(sample[8])
            all_sentences += sentences
        all_para_words = []
        # print(all_sentences[0])
        all_sentences_1 = [sen.encode('utf-8') for sen in all_sentences]

        for sentence in all_sentences_1:
            # print('aa', sentence)
            if sentence == ' ' or sentence == '':
                all_para_words.append(['PAD'])
            else:
                ws = sentence.decode('utf-8').split()
                all_para_words.append(ws)
        if self.flag.use_elmo:
            para_elmo, mask = self.elmo.batch_to_embeddings(all_para_words)
        else:
            para_elmo = torch.ones([3])

        pad_para_chars = []
        pad_para_ids = []
        para_seq_len = []
        max_len = 0
        # padding
        for i in range(6):
            sentence_length = [len(one[6][i]) for one in selected_samples]
            para_seq_len.append(sentence_length)
            lens = max(sentence_length)
            if lens > max_len:
                max_len = lens
        for i in range(6):
            # ids padding
            pad_sentences = [self._padding(one[7][i], max_len) for one in selected_samples]
            # char padding
            pad_char = []
            for sample in selected_samples:
                char = np.array(sample[8][i])
                char = np.reshape(char, [-1, 27])
                while char.shape[0] != max_len:
                    char = np.concatenate([char, np.zeros([1, 27])], 0)
                pad_char.append(char)
            pad_para_chars.append(np.array(pad_char))
            pad_para_ids.append(np.array(pad_sentences))

        pad_para_chars = np.array(pad_para_chars)
        pad_para_ids = np.array(pad_para_ids)
        para_seq_len = np.array(para_seq_len)
        para_seq_len = np.transpose(para_seq_len, [1, 0])
        pad_para_ids = np.transpose(pad_para_ids, [1, 0, 2])
        pad_para_chars = np.transpose(pad_para_chars, [1, 0, 2, 3])
        pad_para_elmo = para_elmo.cpu()
        print('para_elmo', pad_para_elmo.shape)

        return pad_para_chars, pad_para_ids, para_seq_len, pad_para_elmo

