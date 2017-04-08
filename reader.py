import numpy as np

def read_file(f_name):
    return [l.strip() for l in open(f_name, 'r')]

def dup(x): return [x[0]] * int(x[1])
    
def read_instance(ls):
    tmp = ls.split()
    label = tmp[0]
    feature_list = []
    for f in tmp[1:]: feature_list.extend(dup(f.split(':')))
    return (int(label), np.array(feature_list, dtype=np.int32))

def read_data(ls, feature_size=-1):
    vocab_size = 0
    sls = set(ls)
    label, feature_list = [], []
    for _ls in sls:
        _label, _feature_list = read_instance(_ls)
        label.append(_label)
        feature_list.append(_feature_list)
        vocab_size = max(_feature_list.max(), vocab_size)

    if feature_size < 0: feature_size = max([len(fl) for fl in feature_list])
    feature_list_pad = np.zeros((len(sls), feature_size), dtype=np.int32)
    size_list = np.zeros(len(sls), dtype=np.int32)
    for i, fl in enumerate(feature_list):
        l = min(len(fl), feature_size)
        feature_list_pad[i][:l] = np.array(fl[:feature_size])
        size_list[i] = l
    input_label = (np.array(label, dtype=np.int32) + 1) / 2
    return (input_label, feature_list_pad, size_list), vocab_size + 1, feature_size

def read_dataset(f_name=["train.txt", "devel.txt", "test.txt"]):
    f_name = {k:fn for k, fn in zip(["train", "dev", "test"], f_name)}
    train_data, vocab_size, feature_size = read_data(read_file(f_name["train"]))
    dev_data, _, _ = read_data(read_file(f_name["dev"]), feature_size=feature_size)
    test_data, _, _ = read_data(read_file(f_name["test"]), feature_size=feature_size)
    return train_data, dev_data, test_data, vocab_size, feature_size

