import torch
import gensim


w2v_model_path = '/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/basic_data/wordEmbeddings/Word2Vec/GoogleNews-vectors-negative300.bin'

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True)

a = torch.ones(3, 300)
for i, w in enumerate(w2v_model.vocab):
    if i >= 3:
        break
    a[i] = torch.from_numpy(w2v_model.get_vector(w).copy())

print(a)