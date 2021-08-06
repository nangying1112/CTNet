import pickle
import numpy as np

path = 'processed_data/'
pair_num = 3

def get_vocab2id(dataset_path):
    '''
    :param dataset_path: [train_path, dev_path, test_path]
    :return: cocab
    '''
    vocab2id = {'<PAD>': 0, ' ': 1}
    all_data = []
    for path in dataset_path:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            this_set = data['imp'] + data['exp']
            all_data += this_set
    print('The total number of all dataset is %d' % (len(all_data)))
    for sample in all_data:
        # pp.pprint(sample)
        para = sample['para']
        for sentence in para:
            words = sentence[0].split() + sentence[1].split()
            for word in words:
                if word not in vocab2id.keys():
                    vocab2id[word] = len(vocab2id.keys())
        pair = sample['pair']
        for sentence in pair:
            words = sentence.split()
            for word in words:
                if word not in vocab2id.keys():
                    vocab2id[word] = len(vocab2id.keys())
    print(len(vocab2id))
    print(vocab2id)
    with open('exp_imp_vocab2id', 'wb') as f:
        pickle.dump(vocab2id, f, pickle.HIGHEST_PROTOCOL)
    return vocab2id


def get_char2id(vocab2id):
    '''
    :param dataset_path:
    :return:
    '''
    char2id = {}
    vocab = vocab2id.keys()
    max_word_length = max([len(w) for w in vocab])
    for word in vocab:
        for ch in word:
            if ch not in char2id.keys():
                char2id[ch] = len(char2id)
    print(len(char2id))
    return char2id, max_word_length



def pair2ids(sentences, vocab2id):
    '''
    :param sentences: list, [arg1, arg2]; arg1, arg2: list
    :param vocab: dict
    :return: ids (list)
    '''
    pair_ids = []
    for sentence in sentences:
        # print(sentence)
        sentence_ids = []
        for word in sentence.split():
            sentence_ids.append(vocab2id[word])
        pair_ids.append(sentence_ids)
    return pair_ids

def para2ids(sentences, vocab2id):
    '''
    :param sentences: list, [arg1, arg2]; arg1, arg2: list
    :param vocab: dict
    :return: ids (list)
    '''
    para_ids = []
    for sentence in sentences:
        sentence_ids = []
        for word in sentence.split():
            sentence_ids.append(vocab2id[word])
        para_ids.append(sentence_ids)
    return para_ids

def pair2char(sentences, char2id):
    sentences_ids = []
    for sentence in sentences:
        words = sentence.split()
        # print('a', max_word_length)
        sentence_char = np.zeros([len(words), max_word_length])
        for i in range(len(words)):
            for j in range(len(words[i])):
                sentence_char[i][j] = char2id[words[i][j]]
        sentences_ids.append(sentence_char)
    return sentences_ids

def para2char(sentences, char2id):
    sentences_char = []
    for sentence in sentences:
        words = sentence.split()
        sen_char = np.zeros([len(words), max_word_length])
        for i in range(len(words)):
            for j in range(len(words[i])):
                sen_char[i][j] = char2id[words[i][j]]
        sentences_char.append(sen_char)
    return sentences_char

if __name__ == "__main__":


    keys = ['train', 'valid', 'test']

    dataset_path = ['processed_data/train_exp_imp_6sen_clean', 'processed_data/valid_exp_imp_6sen_clean', 'processed_data/test_exp_imp_6sen_clean']
    with open('processed_data/word2id', 'rb') as f:
        vocab2id = pickle.load(f)
    char2id, max_word_length = get_char2id(vocab2id)
    train_dev_test_ids_char = []
    for key in keys:

        with open(path+key+'_exp_imp_6sen_clean', 'rb') as f:
            data = pickle.load(f)
            conn_adj = np.load('adjacent_matrix/'+key+'_conn_adj.npy')
            core_adj = np.load('adjacent_matrix/' + key + '_core_adj.npy')
            word_adj = np.load('adjacent_matrix/' + key + '_word_adj.npy')
            final_adj = np.load('adjacent_matrix/' + key + '_final_adj.npy')
            diag = np.diag(np.ones(2 * pair_num))

            adj_to_process = [conn_adj, core_adj, word_adj]
            for one in adj_to_process:
                for dim in range(2):
                    print(len(one[dim]))
                    for i in range(len(one[dim])):
                        # print('before', one[dim][i])
                        one[dim][i] = one[dim][i] + diag
                        one[dim][i] = one[dim][i] > 0
                        one[dim][i] = one[dim][i].astype(int)
                        # print('after', one[dim][i])
            conn_dict = {'exp': conn_adj[0], 'imp': conn_adj[1]}
            core_dict = {'exp': core_adj[0], 'imp': core_adj[1]}
            word_dict = {'exp': word_adj[0], 'imp': word_adj[1]}
            final_dict = {'exp': final_adj[0], 'imp': final_adj[1]}

            dr_types = ['imp', 'exp']
            for drt in dr_types:
                assert len(data[drt]) == len(conn_dict[drt]) == len(core_dict[drt])
                for j in range(len(data[drt])):
                    assert data[drt][j]['pair'][1] == data[drt][j]['para'][-1]
                    # print()
                    data[drt][j]['pair_ids'] = pair2ids(data[drt][j]['pair'], vocab2id)
                    data[drt][j]['para_ids'] = para2ids(data[drt][j]['para'], vocab2id)
                    data[drt][j]['pair_char'] = pair2char(data[drt][j]['pair'], char2id)
                    data[drt][j]['para_char'] = para2char(data[drt][j]['para'], char2id)
                    data[drt][j]['conn_adj'] = conn_dict[drt][j]
                    data[drt][j]['core_adj'] = core_dict[drt][j]
                    data[drt][j]['word_adj'] = word_dict[drt][j]
                    data[drt][j]['final_adj'] = final_dict[drt][j]
                    print(data[drt][j]['label'])
                    print(data[drt][j]['way11'])
                    print()
                    assert data[drt][j]['word_adj'][2][2] == 1

            train_dev_test_ids_char.append(data)

    with open('processed_data/train_exp_para_adj', 'wb') as f:
        pickle.dump(train_dev_test_ids_char[0]['exp'], f, pickle.HIGHEST_PROTOCOL)
    with open('processed_data/train_imp_para_adj', 'wb') as f:
        pickle.dump(train_dev_test_ids_char[0]['imp'], f, pickle.HIGHEST_PROTOCOL)
    with open('processed_data/dev_para_adj', 'wb') as f:
        pickle.dump(train_dev_test_ids_char[1], f, pickle.HIGHEST_PROTOCOL)
    with open('processed_data/test_para_adj', 'wb') as f:
        pickle.dump(train_dev_test_ids_char[2], f, pickle.HIGHEST_PROTOCOL)




