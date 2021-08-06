import pickle
import numpy as np

path = 'adjacent_matrix/'
if __name__ == "__main__":

    pair_num = 3
    dataset = ['train', 'valid', 'test']

    for key in dataset:
        with open('processed_data/'+key+'_exp_imp_6sen', 'rb') as f:
            docs = pickle.load(f)
        types = ['exp', 'imp']
        exp_imp_conn_adjs = []
        for ty in types:
            conn_adjs = []
            for doc in docs[ty]:
                init_adj = np.zeros([2 * pair_num, 2 * pair_num])
                sentences = doc['para']
                conns = doc['para_conns']
                assert len(sentences) == len(conns)+1
                print(conns)
                for j in range(len(conns)):
                    if conns[j] != "NULL":
                        init_adj[j][j+1] += 1
                        init_adj[j+1][j] += 1
                conn_adjs.append(init_adj)
            exp_imp_conn_adjs.append(conn_adjs)
        np.save(path+key+'_conn_adj', exp_imp_conn_adjs)



