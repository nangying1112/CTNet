import pickle
import numpy as np

path = 'adjacent_matrix/'

if __name__ == "__main__":

    pair_num = 3
    dataset = ['train', 'valid', 'test']
    for key in dataset:
        with open(path+key+'_chains.pkl', 'rb') as f:
            all_chains = pickle.load(f)

        with open(path+key+'_no_chains.pkl', 'rb') as f:
            no_chains = pickle.load(f)

        with open('processed_data/'+key+'_exp_imp_6sen', 'rb') as f:
            docs = pickle.load(f)

        doc_for_core = {'exp':[], 'imp':[]}

        exp_imp_core_adjs = []
        print(len(all_chains['imp']))
        print(no_chains['imp'])
        print(len(docs['imp']))
        types = ['exp', 'imp']

        for ty in types:
            core_adjs = []
            for doc in docs[ty]:
                sentences = 0
                for pair in doc:
                    if len(pair[0]) == 0 or len(pair[1]) == 0:
                        continue
                    sentences += 2
                init_adj = np.zeros([2*pair_num, 2*pair_num])
                core_adjs.append(init_adj)
            # print(len(core_adjs))
            # print(core_adjs[2].shape)
            print(all_chains[ty][2])

            chains_docs_num = len(all_chains[ty])
            chain_id = 0

            for i in range(len(docs[ty])):
                if i in no_chains[ty]:
                    continue
                for chain in all_chains[ty][chain_id]:
                    for m in range(len(chain)):
                        for n in range(m + 1, len(chain)):
                            # print(chain[m], chain[n])
                            core_adjs[i][chain[m]][chain[n]] = 1.
                            core_adjs[i][chain[n]][chain[m]] = 1.
                chain_id += 1
            exp_imp_core_adjs.append(core_adjs)
        print(exp_imp_core_adjs[0][2])
        np.save(path+key+'_core_adj', exp_imp_core_adjs)




