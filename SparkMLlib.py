## one hot encoding and Sparse vectors

# sample OHE creation and sparse representation

def one_hot_encoding(raw_feats, ohe_dict_broadcast, num_ohe_feats):
    #return SparseVector(num_ohe_feats, [(ohe_dict_broadcast[(featID, value)],1) for (featID, value) in raw_feats])
    retObj = SparseVector(num_ohe_feats, [(ohe_dict_broadcast.value[ftrTpl], 1.) for ftrTpl in raw_feats])
    return retObj
# Calculate the number of features in sample_ohe_dict_manual
num_sample_ohe_feats = len(sample_ohe_dict_manual.keys())
sample_ohe_dict_manual_broadcast = sc.broadcast(sample_ohe_dict_manual)

# Run one_hot_encoding() on sample_one.  Make sure to pass in the Broadcast variable.
sample_one_ohe_feat = one_hot_encoding(sample_one, sample_ohe_dict_manual_broadcast, num_sample_ohe_feats)

print sample_one_ohe_feat




