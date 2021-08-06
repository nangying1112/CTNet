import pickle
from utils import pp


if __name__ == "__main__":

    dataset_path = ['slstm_data/train_docs_v2_clean', 'slstm_data/valid_docs_v2_clean', 'slstm_data/test_docs_v2_clean']

    conn2id = {}
    conn2num = {}
    conn2relation = {}
    rel2conn = {'Expansion': [], 'Contingency': [], 'Comparison': [], 'Temporal': []}
    for path in dataset_path:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            dr_type = ['exp', 'imp']
            for drt in dr_type:
                for sample in data[drt]:
                    label = sample['label']
                    conn = sample['conn'].lower()

                    if conn not in conn2id.keys():
                        conn2id[conn] = len(conn2relation)

                    if conn not in conn2num.keys():
                        conn2num[conn] = 1
                    else:
                        conn2num[conn] += 1

                    if conn not in conn2relation.keys():
                        conn2relation[conn] = []
                        for l in label:
                            conn2relation[conn].append(l)
                    else:
                        for l in label:
                            if l not in conn2relation[conn]:
                                conn2relation[conn].append(l)

                    for l in label:
                        if conn not in rel2conn[l]:
                            rel2conn[l].append(conn)

    assert len(conn2id) == len(conn2relation) == len(conn2num)

    # pp.pprint(conn2relation)
    print(len(conn2id))
    # pp.pprint(rel2conn)

    rel2id = {'Expansion': 0, 'Contingency': 1, 'Comparison': 2, 'Temporal': 3}
    rel2conn_id = {0:[], 1:[], 2:[], 3:[]}
    for one in rel2conn:
        for c in rel2conn[one]:
            rel2conn_id[rel2id[one]].append(conn2id[c])
    pp.pprint(rel2conn_id)


    with open('conn2num', 'wb') as f:
        pickle.dump(conn2num, f, pickle.HIGHEST_PROTOCOL)


    with open('conn2id', 'wb') as f:
        pickle.dump(conn2id, f, pickle.HIGHEST_PROTOCOL)

    with open('conn2relation', 'wb') as f:
        pickle.dump(conn2relation, f, pickle.HIGHEST_PROTOCOL)

    with open('rel2conn_id', 'wb') as f:
        pickle.dump(rel2conn_id, f, pickle.HIGHEST_PROTOCOL)