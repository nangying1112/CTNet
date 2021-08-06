import pickle
from nltk.corpus import stopwords


if __name__ == "__main__":

    dataset = ['train', 'valid', 'test']

    for key in dataset:
        with open(key + '_exp_imp_3', 'rb') as f:
            docs = pickle.load(f)
            types = ['exp', 'imp']
            for ty in types:
                for doc in docs[ty]:
                    sentences = []
                    para_conns = []
                    dict = {}
                    index = {}
                    chain_index = []
                    last_sentence = " "
                    for i in range(len(doc['para'])):
                        if i == 0:
                            sentences.append(doc['para'][i][0])
                            sentences.append(doc['para'][i][1])
                            last_sentence = doc['para'][i][1]
                            para_conns.append(doc['para_conns'][i])
                        elif doc['para'][i][0] != last_sentence:
                            para_conns.append('NULL')
                            sentences.append(doc['para'][i][0])
                            sentences.append(doc['para'][i][1])
                            last_sentence = doc['para'][i][1]
                            para_conns.append(doc['para_conns'][i])
                        elif doc['para'][i][0] == last_sentence:
                            sentences.append(doc['para'][i][1])
                            last_sentence = doc['para'][i][1]
                            para_conns.append(doc['para_conns'][i])
                    print(len(doc['para']), doc['para'])
                    print(len(sentences), sentences)
                    print(len(para_conns), para_conns)
                    print(doc['para_conns'])
                    assert len(para_conns) == len(sentences) - 1
                    while len(sentences) < 6:
                        sentences = [' ']+sentences
                    while len(para_conns) < 5:
                        para_conns = ['NULL'] + para_conns

                    doc['para'] = sentences
                    doc['para_conns'] = para_conns
                    assert len(para_conns) == len(sentences)-1
                    print()

            with open(key + '_exp_imp_6sen', 'wb') as f:
                pickle.dump(docs, f, pickle.HIGHEST_PROTOCOL)

