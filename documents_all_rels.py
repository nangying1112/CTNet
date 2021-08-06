import os
# import stanfordnlp
import pickle

path = "PDTB 2.0/data/"
dirs = os.listdir(path)
print(dirs)
dirs = dirs[22:24]  # 1~3 val 3~22 trn 22~24 test
print(dirs)
rel_list = ["____Explicit____", "____Implicit____"]
all_rels = ["____Explicit____", "____Implicit____", "____EntRel____", "____NoRel____", "____AltLex____"]
other_rels = ["____EntRel____", "____NoRel____", "____AltLex____"]

samples = []
for dir in dirs:
    file_path = os.path.join(path, dir)
    docs = os.listdir(file_path)
    # print(len(docs))
    for doc in docs:
        doc_path = os.path.join(file_path, doc)
        # print(doc_path)
        with open(doc_path, 'r', encoding='utf-8') as f:

            one_sample = {}
            one_sample['pairs'] = []
            one_sample['exp_imp'] = []
            one_sample['conns'] = []
            one_sample['labels'] = []

            lines = [line.strip() for line in f.readlines()]
            i = 0
            # 遍历行数直到找到一个关系
            while lines[i] not in all_rels:
                i = i+1
                if i >= len(lines)-1:
                    break
            if i == len(lines)-1:
                continue
            #如果是 exp 或者 imp

            pair = []
            # 找到一个 exp 或者 imp 后
            while lines[i] in all_rels:
                cur_exp_imp = lines[i][4:-4]
                one_sample['exp_imp'].append(lines[i][4:-4])
                i = i+1
                sup_flag = 0
                arg1_flag = 0
                while lines[i] not in all_rels:
                    if lines[i] == "____Sup1____":
                        sup_flag = 1
                        # print(lines[i-1])
                        one_sample['labels'].append(lines[i - 1])

                    if lines[i] == "____Arg1____":
                        arg1 = ""
                        if len(lines[i+4]) == 0:
                            # print(doc_path)
                            arg1_flag = 1
                        if sup_flag == 0:
                            # print('no_sup', lines[i-1])
                            # print(doc_path)
                            one_sample['labels'].append(lines[i-1])

                        while lines[i] != '____Arg2____':
                            if lines[i] == '#### Text ####':
                                # print(lines[i+1])
                                # print(lines[i+1][-1])
                                if lines[i+1][-1] == '.':
                                    arg1 = arg1+lines[i+1][0:-1]
                                else:
                                    arg1 = arg1 + lines[i + 1]
                                arg1 += ' . '
                            i = i+1
                        pair.append(arg1)

                    if lines[i] == "____Arg2____":
                        arg2 = ""
                        while lines[i] != '________________________________________________________':
                            if lines[i] == '#### Text ####':
                                if lines[i+1][-1] == '.':
                                    arg2 = arg2+lines[i+1][0:-1]
                                else:
                                    arg2 = arg2+lines[i+1]
                                arg2 += ' . '
                            i = i+1

                        pair.append(arg2)
                        one_sample['pairs'].append(pair)
                    if i >= len(lines)-1:
                        break
                    i = i+1
                if arg1_flag == 1:
                    one_sample['pairs'] = one_sample['pairs'][:-1]
                    one_sample['labels'] = one_sample['labels'][:-1]
                    one_sample['exp_imp'] = one_sample['exp_imp'][:-1]
                    one_sample['conns'] = one_sample['conns'][:-1]
                    arg1_flag = 0
                    # print(one_sample)
                if i == len(lines)-1:
                    break
                # 处理完上一个论元和连接词等等后，遍历找到下一个exp或者imp
                while lines[i] not in all_rels:
                    if lines[i] in other_rels:
                        if len(one_sample['pairs']) != 0:
                            samples.append(one_sample)
                            one_sample = {}
                            one_sample['pairs'] = []
                            one_sample['exp_imp'] = []
                            one_sample['conns'] = []
                            one_sample['labels'] = []
                    i = i+1
                    if i >= len(lines)-1:
                        break
                if i >= len(lines)-1:
                    break
                pair = []
            if len(one_sample['pairs']) != 0:
                samples.append(one_sample)


no_11 = 0
way11 = ['Temporal.Asynchronous', 'Temporal.Synchrony',
         'Contingency.Cause', 'Contingency.Pragmatic cause',
         'Comparison.Contrast', 'Comparison.Concession',
         'Expansion.Conjunction', 'Expansion.Instantiation', 'Expansion.Restatement', 'Expansion.Alternative', 'Expansion.List']
conns_all_docs = []
labels_all_docs = []
way11_all_docs = []
for one_doc in samples:
    conns_one_doc = []
    labels_one_doc = []
    way11_one_doc = []
    for conn_label in one_doc['labels']:
        connect = ""
        label = []
        w = []
        i = 0
        if ',' not in conn_label and 'Expansion' not in conn_label and 'Comparison' not in conn_label and 'Temporal' not in conn_label and 'Contingengy' not in conn_label:
            labels_one_doc.append(['NULL'])
            conns_one_doc.append('NULL')
            way11_one_doc.append('NULL')
            continue
        # print(conn_label)
        if ',' in conn_label:
            while conn_label[i] != ',':
                connect += conn_label[i]
                i = i+1
            conns_one_doc.append(connect)
        else:
            conns_one_doc.append('NULL')
        if 'Expansion' in conn_label:
            label.append('Expansion')
        if 'Comparison' in conn_label:
            label.append('Comparison')
        if 'Contingency' in conn_label:
            label.append('Contingency')
        if 'Temporal' in conn_label:
            label.append('Temporal')

        for key in way11:
            if key in conn_label:
                w.append(key)
        if len(w) == 0:
            no_11 += 1
            # print(conn_label)

        labels_one_doc.append(label)
        way11_one_doc.append(w)
        # print(label)
        # print(w)
        # print(labels_one_doc)
        # print(way11_one_doc)
        # print()

        assert len(labels_one_doc) == len(way11_one_doc)
    one_doc['labels'] = labels_one_doc
    one_doc['conns'] = conns_one_doc
    one_doc['way11'] = way11_one_doc

print(len(samples))
# print(samples[-1])
# for i in range(len(samples)):
#     if 'unless' in samples[i]['conns']:
        # print(samples[i])

# print(no_11)
with open('processed_data/test_docs_all_rels', 'wb') as f:
    pickle.dump(samples, f, pickle.HIGHEST_PROTOCOL)


