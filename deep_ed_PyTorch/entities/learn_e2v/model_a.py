# **YD** the model to train entity embeddings

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.utils import split_in_words
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID
from deep_ed_PyTorch.words.w2v import W2V
from deep_ed_PyTorch.words.w_freq import WFreq

from deep_ed_PyTorch.utils import utils


class ModelA(nn.Module):
    def __init__(self, args, test=False):
        super(ModelA, self).__init__()

        self.args = args

        # **YD** redundant class loading can be improved to pass on args
        """
        self.ent_name_id = EntNameID(args)
        self.w2v = W2V(args)
        self.w_freq = WFreq(args)
        """

        if test:
            # **YD** necessary parameters for the model
            self.args.type = 'double'
            self.args.batch_size = 7
            self.args.num_words_per_ent = 2
            self.args.num_neg_words = 25
            self.args.loss = 'maxm'
            self.args.word_vecs_size = 5
            self.args.ent_vecs_size = 5

            # entity embedding table
            self.entity_embedding = nn.Embedding(100, self.args.ent_vecs_size)
            print(self.entity_embedding.weight)

        else:
            if hasattr(args, 'ent_name_id'):
                self.ent_name_id = args.ent_name_id
            else:
                self.ent_name_id = args.ent_name_id = EntNameID(args)

            if hasattr(args, 'w2v'):
                self.w2v = args.w2v
            else:
                self.w2v = args.w2v = W2V(args)

            if hasattr(args, 'w_freq'):
                self.w_freq = args.w_freq
            else:
                self.w_freq = args.w_freq = WFreq(args)

            # -- Init ents vectors

            # **YD** "get_total_num_ents" has been implemented
            print('\n==> Init entity embeddings matrix. Num ents = ', self.ent_name_id.get_total_num_ents())
            self.entity_embedding = nn.Embedding(self.ent_name_id.get_total_num_ents() + 1, self.args.ent_vecs_size)

            # -- Zero out unk_ent_thid vector for unknown entities.

            # **YD** "unk_ent_thid" has been implemented
            self.entity_embedding.weight[torch.tensor(self.ent_name_id.unk_ent_thid)] = 0

            # -- Init entity vectors with average of title word embeddings.
            # -- This would help speed-up training.
            if args.init_vecs_title_words:
                print('Init entity embeddings with average of title word vectors to speed up learning.')

                # **YD** "get_ent_name_from_wikiid", "get_wikiid_from_thid" has been implemented
                # "split_in_words" "contains_w" has been implemented
                for ent_thid in range(1, self.ent_name_id.get_total_num_ents() + 1):
                    init_ent_vec = torch.zeros(self.args.ent_vecs_size)
                    ent_name = self.ent_name_id.get_ent_name_from_wikiid(
                        self.ent_name_id.get_wikiid_from_thid(ent_thid)
                    )

                    # **YD** for debugging purpose
                    # print(str(ent_thid), self.ent_name_id.get_wikiid_from_thid(str(ent_thid)), ent_name)
                    # print(ent_name, type(ent_name))
                    words_plus_stop_words = split_in_words(ent_name)
                    num_words_title = 0
                    for w in words_plus_stop_words:
                        if self.w_freq.contains_w(w):
                            # **YD** twisted function calling
                            init_ent_vec += self.w2v.M(
                                torch.tensor(
                                    self.w_freq.get_id_from_word(w)
                                )
                            ).clone()
                            num_words_title += 1

                    if num_words_title > 0:
                        assert init_ent_vec.norm() > 0, ent_name
                    init_ent_vec /= num_words_title

                    if init_ent_vec.norm() > 0:
                        self.entity_embedding.weight[torch.tensor(ent_thid)] = init_ent_vec.clone()

            self.entity_embedding = nn.Embedding.from_pretrained(self.entity_embedding.weight, freeze=False)
            # self.entity_embedding.weight.detach_()

    """
    def forward(self, ent_th_id, word_vec):
        '''
        :param ent_th_id: entity th id (batch_size)
        :param word_vec: word vectors (batch_size*num_words_per_ent*num_neg_words, ent_vecs_size)
        :return: cosine similarity matrix between entities embedding and word embedding
        '''
        word_vec = F.normalize(word_vec, 2)
        word_vec = word_vec.view(self.args.batch_size,
                                 self.args.num_words_per_ent * self.args.num_neg_words,
                                 self.args.ent_vecs_size)
        ent_emb = self.entity_embedding(ent_th_id)
        print('before normalize', ent_emb)
        ent_emb = F.normalize(ent_emb, 2)
        print('after normalize', ent_emb)
        ent_emb = ent_emb.view(self.args.batch_size,
                               1,
                               self.args.ent_vecs_size)

        result = torch.bmm(word_vec, torch.transpose(ent_emb, 1, 2)).squeeze(2)
        print(result.shape)
        assert len(result.shape) == 2 and result.shape[0] == self.args.batch_size \
            and result.shape[1] == self.args.num_words_per_ent * self.args.num_neg_words

        return result
    """

    def forward(self, inputs):
        """
        :param inputs = [
            [word_ids, word_vec, word_freq_power]
            [ent_component_words],
            [ent_thids, ent_wikiids],
        ]

            word_ids: shape = [batch_size * num_words_per_ent * num_neg_words],
                for each entity (each training instance), generate "num_words_per_ent" context positive words, for each
                positive word, generate "num_neg_words" to classify the positive word from all the words.

            word_vec: shape = [batch_size * num_words_per_ent * num_neg_words, ent_vecs_size],
                corresponding word vector for each word_id in word_ids

            word_freq_power: shape = [batch_size * num_words_per_ent * num_neg_words],
                frequence ** power for each word in word_ids

            ent_component_words: shape = [batch_size * num_words_per_ent]
                the positive word w_id for each generated "num_words_per_ent"

            ent_thids: shape = [batch_size]
                thids for entities,

            ent_wikiids: shape = [batch_size]
                wikiids for entities,
        :return: cosine similarity matrix between entities embedding and word embedding
        """
        word_vec = inputs[0][1]
        ent_th_id = inputs[2][0]

        word_vec = F.normalize(word_vec, 2)
        word_vec = word_vec.view(self.args.batch_size,
                                 self.args.num_words_per_ent * self.args.num_neg_words,
                                 self.args.ent_vecs_size)
        ent_emb = self.entity_embedding(ent_th_id)
        # print('before normalize', ent_emb)
        ent_emb = F.normalize(ent_emb, 2)
        # print('after normalize', ent_emb)
        ent_emb = ent_emb.view(self.args.batch_size,
                               1,
                               self.args.ent_vecs_size)

        result = torch.bmm(word_vec, torch.transpose(ent_emb, 1, 2)).squeeze(2)

        result = result.view(self.args.batch_size * self.args.num_words_per_ent, self.args.num_neg_words)
        # assert len(result.shape) == 2 and result.shape[0] == self.args.batch_size \
        #       and result.shape[1] == self.args.num_words_per_ent * self.args.num_neg_words

        return result


def test(args):
    model = ModelA(args, test=True)
    model.cuda()
    model.train()

    word_ids = torch.ones(args.batch_size * args.num_words_per_ent * args.num_neg_words, dtype=torch.long)
    word_vec = torch.ones(args.batch_size * args.num_words_per_ent * args.num_neg_words, args.word_vecs_size)
    word_freq_power = torch.randn(args.batch_size * args.num_words_per_ent * args.num_neg_words)

    ent_component_words = torch.ones(args.batch_size * args.num_words_per_ent, dtype=torch.long)

    ent_thids = torch.ones(args.batch_size, dtype=torch.long)
    ent_wikiids = torch.ones(args.batch_size, dtype=torch.long)

    inputs = [
        [word_ids, word_vec, word_freq_power],
        [ent_component_words],
        [ent_thids, ent_wikiids],
    ]

    inputs = utils.move_to_cuda(inputs)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    #word_vec = word_vec.to(device)
    #ent_th_id = ent_th_id.to(device)
    out = model(inputs)
    loss = torch.sum(out)
    print('Forward Success!')
    loss.backward()
    print('Backward Success!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test entity embedding model_a',
        allow_abbrev=False,
    )
    args = parser.parse_args()
    test(args)