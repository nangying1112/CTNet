import pickle
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np

path = 'adjacent_matrix/'
if __name__ == "__main__":

    swords = stopwords.words('english')
    signs = ['(', ')', '?', ';', '!', '/', ',', '[', ']', '<', '>', '"', '@', ':', '…', '“', '$', '-', '+', '=', '*',
             '~', '#', '&', '\\', '¤', '_', '^', '{', '}', '|', '`', '.', '']
    swords = swords+signs
    print(swords)
    pair_num = 3

    dataset = ['train', 'valid', 'test']

    for key in dataset:
        exp_imp_word_adj = []
        words_index = {}
        words_index['exp'] = []
        words_index['imp'] = []
        with open('processed_data/'+key+'_exp_imp_6sen', 'rb') as f:
            exp_imp_docs = pickle.load(f)
            print(len(exp_imp_docs))
        exp_docs = exp_imp_docs['exp']
        imp_docs = exp_imp_docs['imp']
        types = ['exp', 'imp']
        for ty in types:
            word_adj = []
            for doc in exp_imp_docs[ty]:
                init_adj = np.zeros([2 * pair_num, 2 * pair_num])
                sentences = doc['para']
                assert len(sentences) == 6
                dict = {}
                index = {}
                chain_index = []

                for i in range(len(sentences)):
                    words = sentences[i].split(' ')
                    for j in range(len(words)):
                        if words[j].lower() in swords:
                            continue
                        if words[j] not in dict.keys():
                            dict[words[j]] = [i]
                            index[words[j]] = [(i, j)]
                        else:
                            dict[words[j]].append(i)
                            index[words[j]].append((i, j))
                        wi_synset = wn.synsets(lemmatizer.lemmatize(words[j].lower()))
                        for l in range(len(wi_synset)):
                            wi_lemmas = wi_synset[l].lemmas()
                            for lemma in wi_lemmas:
                                if lemma.name().lower() == words[j].lower() :
                                    continue
                                if lemma.name() in dict.keys():
                                    # print(words[j], lemma.name())
                                    dict[lemma.name()].append(i)
                                    index[lemma.name()].append((i, j))
                this_index = {}
                for word_chain in dict.keys():

                    if len(dict[word_chain]) > 1:
                        this_index[word_chain] = index[word_chain]
                        # print(word_chain, dict[word_chain])

                        for j in range(len(dict[word_chain])):
                            for k in range(j, len(dict[word_chain])):
                                # print(dict[word_chain][j], dict[word_chain][j])
                                init_adj[dict[word_chain][j]][dict[word_chain][j]] = 1
                print(this_index)
                print(sentences)
                print()
                for hh in this_index:
                    one = this_index[hh][0]
                words_index[ty].append(this_index)
                word_adj.append(init_adj)
            exp_imp_word_adj.append(word_adj)
            print(len(exp_imp_word_adj), len(words_index['exp']))
            assert len(exp_imp_word_adj[0]) == len(words_index['exp'])

        np.save(path+key + '_word_adj', exp_imp_word_adj)

        with open(path+key+'_word_chain_index.pkl', 'wb') as f:
            pickle.dump(words_index, f, pickle.HIGHEST_PROTOCOL)