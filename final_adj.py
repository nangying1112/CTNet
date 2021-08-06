import numpy as np
import pickle as pkl

path = 'adjacent_matrix/'
if __name__=="__main__":

    pair_num = 3
    dataset = ['train', 'valid', 'test']


    for key in dataset:
        exp_imp_final_adj = []
        core_adj = np.load(path+key+'_core_adj.npy')
        conn_adj = np.load(path+key+'_conn_adj.npy')
        word_adj = np.load(path+key+'_word_adj.npy')
        print(len(word_adj[0]))
        print(len(conn_adj[0]))
        print(len(core_adj[0]))

        for ei in range(2):
            # with open(path+key+'_core_index.pkl', 'rb') as f:
            #     core_index = pkl.load(f)
            # with open(path+key+'_word_chain_index.pkl', 'rb') as f:
            #     word_chain_index = pkl.load(f)
            # with open(path+key+'_no_chains.pkl', 'rb') as f:
            #     no_chains = pkl.load(f)
            #
            # assert len(core_index) + len(no_chains) == len(word_chain_index)
            diag = np.diag(np.ones(2*pair_num))
            # print(diag)
            final_adj = []
            print(len(core_adj[ei]))
            for i in range(len(core_adj[ei])):
                fh = core_adj[ei][i]+conn_adj[ei][i]+word_adj[ei][i]+diag
                fh = fh > 0
                fh = fh.astype(int)
                final_adj.append(fh)
            print(len(final_adj))
            # for i in range(10):

            #     print(final_adj[i])
            exp_imp_final_adj.append(final_adj)
        np.save('adjacent_matrix/'+key+'_final_adj', exp_imp_final_adj)