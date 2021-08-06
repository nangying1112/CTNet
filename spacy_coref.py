import spacy
import pickle

path = 'adjacent_matrix/'
print(spacy.__version__)
if __name__ == "__main__":

    nlp = spacy.load('en_coref_sm')
    dataset = ['valid', 'test', 'train']
    for dataset_key in dataset:
        with open('processed_data/'+dataset_key+'_exp_imp_6sen', 'rb') as f:
            exp_imp_docs = pickle.load(f)
        types = ['exp', 'imp']

        doc_for_core = {'exp':[], 'imp':[]}
        doc_char_id = {'exp':[], 'imp':[]}
        all_chains = {'exp': [], 'imp': []}
        no_chains_doc = {'exp': [], 'imp': []}

        for ty in types:
            for doc in exp_imp_docs[ty]:
                last_sentence = " "
                paragraph = ""
                char_id = {}
                cur_sentence = 0
                char_id[cur_sentence] = len(paragraph)
                cur_sentence += 1
                for sentence in doc['para']:
                    new_sen = ""
                    for char in sentence:
                        if char == "'" or char == '"' or char == '(' or char == ')':
                            continue
                        else:
                            new_sen += char
                    paragraph += new_sen
                    if new_sen[-1] != '.' or new_sen[-1] != ',' or new_sen[-1] != '?' or new_sen[-1] != '!':
                        paragraph += '. '
                    else:
                        paragraph += ' '
                    char_id[cur_sentence] = len(paragraph)
                    cur_sentence += 1
                print(paragraph)
                print(doc['para'])
                print(char_id)
                print()

                doc_for_core[ty].append(paragraph)
                doc_char_id[ty].append(char_id)
            print(doc_for_core[ty][0])
            print(doc_char_id[ty][0])
            print(nlp(doc_for_core[ty][0])._.coref_clusters)

            for i in range(len(doc_for_core[ty])):
                print(i)
                doc = doc_for_core[ty][i]
                char_id = doc_char_id[ty][i]
                core = nlp(doc)

                if core._.coref_clusters == None:
                    no_chains_doc[ty].append(i)
                    continue
                chains_one_doc = []
                for clust in core._.coref_clusters:
                    clust_id = clust.i  # 该指代在本文档中所有指代中的序号
                    main_mention = clust.main  # 主描述，共指链中的代表词语
                    # print("appearance of cluster %d: %s" % (clust_id, main_mention))
                    chain = []
                    for mention in clust.mentions:
                        # print("\t%s at [%d,%d]" % (mention.text, mention.start_char, mention.end_char))
                        # print(mention.start_char)
                        for key in char_id:
                            if char_id[key] < mention.start_char:
                                continue
                            elif char_id[key] == mention.start_char:
                                chain.append(key)
                            else:
                                if key-1 not in chain:
                                    assert key-1 != -1
                                    chain.append(key-1)
                                break
                    # print(chain)
                    chains_one_doc.append(chain)
                all_chains[ty].append(chains_one_doc)

        print(all_chains['exp'][0])
        print(doc_for_core['exp'][0])
        print(dataset_key)
        print(dataset_key+'_no_chains.pkl')
        with open(path+dataset_key+'_no_chains.pkl', 'wb') as f:
            pickle.dump(no_chains_doc, f, pickle.HIGHEST_PROTOCOL)
        with open(path+dataset_key+'_chains.pkl', 'wb') as f:
            pickle.dump(all_chains, f, pickle.HIGHEST_PROTOCOL)

