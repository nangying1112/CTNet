import pickle
import nltk
from nltk.corpus import stopwords

print(nltk.__version__)

def replaceSignals(sentence):

    sentence = sentence.replace('!!', ' ')
    sentence = sentence.replace('..', ' ')
    sentence = sentence.replace('??', ' ')
    sentence = sentence.replace('\' ', ' ')
    sentence = sentence.replace(' \'\' ', ' ')
    sentence = sentence.replace(' \' ', ' ')
    sentence = sentence.replace('zzz', ' ')
    sentence = sentence.replace('aaa', ' ')
    sentence = sentence.replace('%20', ' ')
    sentence = sentence.replace('\xa0', ' ')
    sentence = sentence.replace('  ', ' ')
    for signal in signs:
        sentence = sentence.replace(signal, ' ' + signal + ' ')
    return sentence


def remove_stopwords():
    test_ids = []
    vocab = {}  # vocab2id
    ivocab = {}  # id2vocab
    vocab['NULL'] = 0
    ivocab[0] = 'NULL'
    with open('test-check-clean.txt', 'a') as f:
        for line in open('test-check.txt', encoding='utf-8'):
            divs = line.split('\t')
            for div_ in divs[:2]:  # e.g. divs = [question, label, 0/1]
                sentence = div_.lower()
                # 去除多余的空格：
                sentence_wo_blanks = ""
                for i in range(len(sentence)):
                    if i == len(sentence)-1:
                        sentence_wo_blanks += sentence[i]
                        continue
                    if sentence[i] != ' ':
                        sentence_wo_blanks += sentence[i]
                    if sentence[i] == ' ' and sentence[i + 1] == ' ':
                        continue
                    if sentence[i] == ' ' and sentence[i + 1] != ' ':
                        sentence_wo_blanks += sentence[i]

                sentence = sentence_wo_blanks
                new_sentence = ""
                words = sentence.split(' ')
                new_words = []
                for word in words:
                    if word in signs:
                        continue
                    else:
                        new_words.append(word)
                for word in new_words:
                    new_sentence = new_sentence + word + ' '
                if len(new_sentence) == 1:
                    new_sentence = 'NULL'
                f.write(new_sentence)
                f.write('\t')
            f.write(divs[-1])

if __name__ == "__main__":

    eng_stopwords = stopwords.words('English')
    print(eng_stopwords)
    eng_stopwords.append(' ')
    signs = ['(', ')', '?', ';', '!', '/', ',', '[', ']', '<', '>', '"', '@', ':', '…', '“', '$', '-', '+', '=', '*',
             '~', '#', '&', '\\', '¤', '_', '^', '{', '}', '|', '`']

    dataset = ['train', 'valid', 'test']

    for key in dataset:
        with open(key+'_docs_all_rels', 'rb') as f:
            docs = pickle.load(f)
        print(docs[-1]['labels'])
        pair_num = 3
        docs_3 = {}
        docs_3['imp'] = []
        docs_3['exp'] = []
        for doc in docs:
            assert len(doc['pairs']) == len(doc['conns']) == len(doc['labels'])
            for i in range(len(doc['pairs'])):
                one_sample = {}
                one_sample['pair'] = doc['pairs'][i]
                one_sample['label'] = doc['labels'][i]
                one_sample['way11'] = doc['way11'][i]
                one_sample['para'] = doc['pairs'][max(0, i-2):i+1]

                assert one_sample['pair'] == one_sample['para'][-1]
                assert len(one_sample['para']) <= 3
                # print(i, max(0, i-3), max(1, i))
                print()
                # print(len(one_sample['para']))
                one_sample['conn'] = doc['conns'][i]
                one_sample['para_conns'] = doc['conns'][max(0, i-2):i+1]
                # print(one_sample['pair'])
                # print(one_sample['label'])
                # print(doc['exp_imp'][i])
                print(one_sample['conn'])
                print(one_sample['para_conns'])
                if doc['exp_imp'][i] == 'Implicit':
                    docs_3['imp'].append(one_sample)
                if doc['exp_imp'][i] == 'Explicit':
                    docs_3['exp'].append(one_sample)
        print(len(docs_3['imp']))
        with open(key+'_exp_imp_3', 'wb') as f:
            pickle.dump(docs_3, f, pickle.HIGHEST_PROTOCOL)
