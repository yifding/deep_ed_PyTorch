# **YD** the model to train deep-ed-global for entity disambiguation
import copy
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.utils import utils
from deep_ed_PyTorch.ed.args import arg_parse


class EntityContext(nn.Module):
    def __init__(self, args):
        super(EntityContext, self).__init__()
        self.args = args

        self.A_linear = nn.Parameter(torch.ones(args.ent_vecs_size))
        self.B_linear = nn.Parameter(torch.ones(args.ent_vecs_size))

    def forward(self, inputs):
        """
        :param inputs = [
            [ctxt_words, ctxt_words_vec],
            [cand_entities, cand_entities_vec],
            [p_e_m],
        ]

            # **YD** batch_size = num_mentions: ED task is to classify the correct entity for a given mention.
            # local_model only looks at mention (surface format), candidate entities, context words and properties of
            # these candidate entities.

            # global_model: considers the ED selection as a sequence classification task, naming to for a given document
            # with multiple mentions, how to find the matched entities for the whole sequence.

            ctxt_words: shape = [num_mentions, ctxt_window]
                for each training instance (considering one mention; global?), the context words within the context
                window.

            ctxt_words_vec: shape = [num_mentions, ctxt_window, word_vecs_size]
                corresponding word embeddings for  each context word

            cand_entities: shape = [num_mentions, max_num_cand]
                truncated candidate entities for mentions, entities are represented by their thid

            cand_entities_vec: shape = [num_mentions, max_num_cand, ent_vecs_size]
                entity embedding for each candidate entities

            p_e_m: shape = [num_mentions, max_num_cand], log(p(e|m))
                for each mention, the candidate entities are found by the p(e|m) in order. It is also used as a feature
                to predict the final local classification score for a mention-entity pair.


        :return: entity_context_sim_scores, beta

            entity_context_sim_scores: shape = [num_mentions, max_num_cand]
                local scores of mention-candidate_entity pairs

            beta: shape = [num_mentions, ctxt_window]
                attention of mention-ctxt_word, for visualization

        """

        [
            [ctxt_words, ctxt_words_vec],
            [cand_entities, cand_entities_vec],
            p_e_m,
        ] = inputs

        # pre_u.shape = [num_mentions, max_num_cand, ent_vecs_size]

        # **YD** infer the first dimension of input candidate entity vectors
        self.args.num_mentions = cand_entities_vec.size(0)
        # print('self.args.num_mentions', self.args.num_mentions)

        pre_u = (
                cand_entities_vec.view(self.args.num_mentions * self.args.max_num_cand, self.args.ent_vecs_size) *
                self.A_linear
        ).view(self.args.num_mentions, self.args.max_num_cand, self.args.ent_vecs_size)

        # assert pre_u.shape == torch.Size([self.args.num_mentions, self.args.max_num_cand, self.args.ent_vecs_size])

        # post_u.shape = [num_mentions, word_vecs_size, ctxt_window]
        post_u = torch.transpose(ctxt_words_vec, 1, 2)
        # assert post_u.shape == torch.Size([self.args.num_mentions, self.args.word_vecs_size, self.args.ctxt_window])

        # u_vec.shape = [num_mentions, max_num_cand, ctxt_window]
        u_vec = torch.bmm(pre_u, post_u)
        # assert u_vec.shape == torch.Size([self.args.num_mentions, self.args.max_num_cand, self.args.ctxt_window])

        # u.shape = [num_mentions, ctxt_window]
        u = torch.max(u_vec, dim=1).values
        # assert u.shape == torch.Size([self.args.num_mentions, self.args.ctxt_window])

        # top_k.shape = [num_mentions, R]
        top_k = torch.topk(u, k=self.args.R).values
        # assert top_k.shape == torch.Size([self.args.num_mentions, self.args.R])

        # top_min.shape = [num_mentions]
        top_min = torch.min(top_k, dim=1).values
        # assert top_min.shape == torch.Size([self.args.num_mentions])

        # **YD** the model autograph may cause problem here, not 100% sure which one to choose
        sketch = top_min.view(self.args.num_mentions, 1).clone()
        # sketch = top_min.view(self.args.num_mentions, 1)

        minus_result = u - sketch

        nn.Threshold(0, -50, True).forward(minus_result)

        minus_result = minus_result + sketch

        # beta.shape = [num_mentions, ctxt_window]
        beta = F.softmax(minus_result, dim=1)

        # assert beta.shape == torch.Size([self.args.num_mentions, self.args.ctxt_window])

        # **YD** second step,

        # ctxt_full_embeddings.shape = [num_mentions, word_vecs_size]
        ctxt_full_embeddings = torch.bmm(
            torch.transpose(ctxt_words_vec, 1, 2),
            beta.view(self.args.num_mentions, self.args.ctxt_window, 1)
        ).squeeze(2)

        # entity_context_sim_scores_pre.shape = [num_mentions, max_num_cand, ent_vecs_size]
        entity_context_sim_scores_pre = cand_entities_vec

        # entity_context_sim_scores_post.shape = [num_mentions, word_vecs_size, 1]
        entity_context_sim_scores_post = (ctxt_full_embeddings * self.B_linear).view(
            self.args.num_mentions, self.args.word_vecs_size, 1
        )

        # entity_context_sim_scores.shape = [num_mentions, max_num_cand]
        entity_context_sim_scores = torch.bmm(
            entity_context_sim_scores_pre,
            entity_context_sim_scores_post
        ).squeeze(2)

        return entity_context_sim_scores, beta


class MessageOneRound(nn.Module):
    def __init__(self, args):
        super(MessageOneRound, self).__init__()
        self.args = args
        self.dummy_param = nn.Parameter(torch.empty(0))

        '''
        self.mask = torch.ones(self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions)
        for i in range(self.args.max_num_cand):
            for j in range(self.args.num_mentions):
                self.mask[j, i, j] = 0
        self.mask.require_grad=False
        if args.type == 'cuda':
            self.mask = self.mask.cuda()
        '''

    def forward(self, unary_plus_pairwise, old_message):
        self.args.num_mentions = unary_plus_pairwise.size(0)

        '''
        assert unary_plus_pairwise.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions, self.args.max_num_cand]
        )
        '''

        '''
        assert old_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        crf_old_message = (1.0 - self.args.lbp_damp) * torch.exp(old_message)
        '''
        assert crf_old_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        # equation (10), add sum old message and unary_plus_pairwise
        crf_new_message = unary_plus_pairwise + \
            old_message.sum(2).repeat(self.args.num_mentions, self.args.max_num_cand, 1, 1)
        '''
        assert crf_new_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions, self.args.max_num_cand]
        )
        '''

        select_crf_new_message = crf_new_message.max(3).values
        '''
        assert select_crf_new_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        select_crf_new_message = F.log_softmax(select_crf_new_message, dim=1)
        '''
        assert select_crf_new_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        # **YD** bugs here..
        # select_crf_new_message.diagonal(dim1=0, dim2=2).fill_(0)

        mask = torch.ones(select_crf_new_message.shape)
        for i in range(select_crf_new_message.size(1)):
            for j in range(select_crf_new_message.size(0)):
                mask[j, i, j] = 0
        mask.require_grad = False
        '''
        if self.args.type == 'cuda':
            mask = mask.cuda()
        '''

        device = self.dummy_param.device
        mask = mask.to(device)

        select_crf_new_message = select_crf_new_message * mask
        '''
        assert select_crf_new_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        select_crf_new_message = torch.exp(select_crf_new_message) * self.args.lbp_damp

        crf_total_message = torch.log(select_crf_new_message + crf_old_message)

        return crf_total_message


class ModelGlobal(nn.Module):
    def __init__(self, args):
        super(ModelGlobal, self).__init__()
        self.args = args
        self.entity_context = EntityContext(args)

        self.C_linear = nn.Parameter(torch.ones(args.ent_vecs_size))
        self.linear1 = nn.Linear(2, self.args.nn_pem_interm_size)
        self.linear2 = nn.Linear(self.args.nn_pem_interm_size, 1)
        self.message_pass_list = [copy.deepcopy(MessageOneRound(args)) for _ in range(self.args.lbp_iter)]

    def forward(self, inputs):
        entity_context_sim_scores, beta = self.entity_context(inputs)

        [
            [ctxt_words, ctxt_words_vec],
            [cand_entities, cand_entities_vec],
            p_e_m,
        ] = inputs

        # check num_mentions match
        self.args.num_mentions = cand_entities_vec.size(0)


        # build entity pairwise similarity by dot product
        entity_pairwise_pre = cand_entities_vec.view(self.args.num_mentions * self.args.max_num_cand,
                                                     self.args.ent_vecs_size) * self.C_linear

        '''
        assert entity_pairwise_pre.shape == torch.Size(
            [self.args.num_mentions * self.args.max_num_cand, self.args.ent_vecs_size]
        )
        '''

        entity_pairwise_post = cand_entities_vec.view(
            self.args.num_mentions * self.args.max_num_cand, self.args.ent_vecs_size
        ).T
        '''
        assert entity_pairwise_post.shape == torch.Size(
            [self.args.ent_vecs_size, self.args.num_mentions * self.args.max_num_cand]
        )
        '''

        entity_pairwise = torch.mm(entity_pairwise_pre, entity_pairwise_post).view(
            self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions,self.args.max_num_cand
            )
        '''
        assert entity_pairwise.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions, self.args.max_num_cand]
        )
        '''

        # build unary score by duplicating context score
        unary = entity_context_sim_scores.repeat(self.args.num_mentions, self.args.max_num_cand, 1, 1)
        '''
        assert unary.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions, self.args.max_num_cand]
        )
        '''

        # sum up the unary and pairwise
        unary_plus_pairwise = unary + entity_pairwise
        assert unary_plus_pairwise.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions, self.args.max_num_cand]
        )

        # initial message is '0', message is defined from a mention to another mention's certain candidate entity
        round_message = unary_plus_pairwise.max(3).values * 0
        '''
        assert round_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        for message_pass_layer in self.message_pass_list:
            round_message = message_pass_layer(unary_plus_pairwise, round_message)
        '''
        assert round_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand, self.args.num_mentions]
        )
        '''

        global_message = round_message.sum(2)
        '''
        assert global_message.shape == torch.Size(
            [self.args.num_mentions, self.args.max_num_cand]
        )
        '''

        final_global_score = F.log_softmax(entity_context_sim_scores + global_message, dim=-1)

        x = torch.cat(
            [
                final_global_score.view(self.args.num_mentions * self.args.max_num_cand, 1),
                p_e_m.view(self.args.num_mentions * self.args.max_num_cand, 1),
            ], dim=1
        )

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = x.view(self.args.num_mentions, self.args.max_num_cand)

        return x, beta, entity_context_sim_scores


def test():
    parser = argparse.ArgumentParser(
        description='test entity embedding model_a',
        allow_abbrev=False,
    )
    args = parser.parse_args()
    args.type = 'cuda'
    args.ctxt_window = 100
    args.R = 25
    args.model = 'local'
    args.nn_pem_interm_size = 100

    args.word_vecs_size = 300
    args.ent_vecs_size = 300
    args.max_num_cand = 6
    args.unk_ent_wikiid = 1
    args.unk_ent_thid = 1
    args.unk_w_id = 1
    args.num_mentions = 13

    args.lbp_iter = 10
    args.lbp_damp = 0.5

    model = ModelGlobal(args)
    model.train()
    model.cuda()


    #  -- ctxt_w_vecs
    ctxt_words = torch.ones(args.num_mentions, args.ctxt_window, dtype=torch.long)
    ctxt_words_vec = torch.rand(args.num_mentions, args.ctxt_window, args.word_vecs_size)

    # -- e_vecs
    cand_entities = torch.ones(args.num_mentions, args.max_num_cand, dtype=torch.long)
    cand_entities_vec = torch.rand(args.num_mentions, args.max_num_cand, args.ent_vecs_size)

    # -- p(e|m)
    p_e_m = torch.zeros(args.num_mentions, args.max_num_cand)

    inputs = [
        [ctxt_words, ctxt_words_vec],
        [cand_entities, cand_entities_vec],
        p_e_m,
    ]

    torch.autograd.set_detect_anomaly(True)
    inputs = utils.move_to_cuda(inputs)
    outputs, beta, entity_context_sim_scores = model(inputs)
    loss = torch.sum(outputs)
    print('Forward Success!')

    loss.backward()
    print('Backward Success!')


if __name__ == '__main__':
    test()