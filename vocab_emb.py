import pickle as pkl
import torch

# build vocab and word2id


signs = ['(', ')', '?', ';', '!', ', ', '[', ']', '<', '>', '"', ': ',  '“', '$', '-', '+', '=', '*',
         '#', '&',   '_', '{', '}', '|']

replace_signs = [' ( ', ' ) ', ' ? ', ' ; ', ' ! ', ' , ', ' [ ', ' ] ', ' < ', ' > ', ' " ', ' : ', ' “ ', ' $ ', ' - ', ' + ', ' = ', ' * ',
          ' # ', ' & ',  ' _ ',  ' { ', ' } ', ' | ']



def replaceSignals(sentence):

    sentence = sentence.replace('can\'t', 'can not')
    sentence = sentence.replace('won\'t', 'will not')
    sentence = sentence.replace('\'ve', ' have')
    sentence = sentence.replace('\'', ' \'')

    for i in range(len(signs)):
        sentence = sentence.replace(signs[i], replace_signs[i])
    # print(sentence.split())
    return sentence


if __name__ == "__main__":

    word_freq = {}
    word_set = set()

    with open('processed_data/train_exp_imp_6sen', 'rb') as f:
        train_docs = pkl.load(f)
    with open('processed_data/test_exp_imp_6sen', 'rb') as f:
        test_docs = pkl.load(f)
    with open('processed_data/valid_exp_imp_6sen', 'rb') as f:
        valid_docs = pkl.load(f)

    for doc in train_docs['exp']:
        print(doc['pair'][0])
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        print(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])


    for doc in train_docs['imp']:
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])

    for doc in valid_docs['exp']:
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])

    for doc in valid_docs['imp']:
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])


    for doc in test_docs['exp']:
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])


    for doc in test_docs['imp']:
        doc['pair'][0] = replaceSignals(doc['pair'][0])
        doc['pair'][1] = replaceSignals(doc['pair'][1])
        for i in range(len(doc['para'])):
            doc['para'][i] = replaceSignals(doc['para'][i])

    with open('processed_data/train_exp_imp_6sen_clean', 'wb') as f:
        pkl.dump(train_docs, f, pkl.HIGHEST_PROTOCOL)
    with open('processed_data/valid_exp_imp_6sen_clean', 'wb') as f:
        pkl.dump(valid_docs, f, pkl.HIGHEST_PROTOCOL)
    with open('processed_data/test_exp_imp_6sen_clean', 'wb') as f:
        pkl.dump(test_docs, f, pkl.HIGHEST_PROTOCOL)

    train_valid_test = train_docs['exp'] + valid_docs['exp'] + test_docs['exp']\
                    + train_docs['imp'] + valid_docs['imp'] + test_docs['imp']

    print(train_docs['exp'][10]['para'])

    for doc in train_valid_test:
        arg1 = doc['pair'][0].split()
        arg2 = doc['pair'][1].split()

        para_word = []
        for sen in doc['para']:
            para_word += sen.split()

        for word in arg1 + arg2 + para_word:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab = ['PAD'] + vocab
    vocab_size = len(vocab)
    print('vocab', vocab_size)
    word2id = dict([(x, y) for (y, x) in enumerate(vocab)])
    print(len(word2id))
    with open('processed_data/word2id', 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)

    with open('processed_data/word2id', 'rb') as f:
        cat2id = pkl.load(f)
    vocab = [key for key in cat2id.keys()]
    print(vocab)
    with open("glove.840B.300d.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    embRec = {}  # w_idx: 1
    emb = torch.randn(len(vocab), 300) * 0.05
    count = 0

    for line in lines:

        vals = line.rstrip("\n").split()
        word = vals.pop(0)
        # print(word)
        if len(vals) != 300:
            continue
        assert len(vals) == 300
        if word in cat2id:
            count += 1
            emb[cat2id[word], :] = torch.FloatTensor([float(v) for v in vals])
    emb = emb.numpy()
    print(count)
    print(emb.shape)
    with open('exp_imp_vocab_embed', 'wb') as f:
        pkl.dump(emb, f, pkl.HIGHEST_PROTOCOL)