import os
import time
import math
import argparse

from itertools import chain
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
dofile 'utils/logger.lua'
dofile 'entities/relatedness/relatedness.lua'  **YD** done, used to represent related valid/test dataset and entity embedding evaluations.
dofile 'entities/ent_name2id_freq/ent_name_id.lua'  
dofile 'words/load_w_freq_and_vecs.lua'
dofile 'words/w2v/w2v.lua'
dofile 'entities/learn_e2v/minibatch_a.lua'
dofile 'entities/learn_e2v/model_a.lua'    
dofile 'entities/learn_e2v/e2v_a.lua'
dofile 'entities/learn_e2v/batch_dataset_a.lua'
"""

from deep_ed_PyTorch.utils import utils

from deep_ed_PyTorch.words.w2v import W2V
from deep_ed_PyTorch.words.w_freq import WFreq
from deep_ed_PyTorch.entities.learn_e2v import E2V
from deep_ed_PyTorch.entities.EX_wiki_words import ExWikiWords
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID
from deep_ed_PyTorch.entities.relatedness import REWTR

from model_a import ModelA
from batch_dataset_a import BatchDatasetA


class LearnA(object):
    def __init__(self, args):
        self.args = args

        if hasattr(args, 'ex_wiki_words'):
            self.ex_wiki_words = args.ex_wiki_words
        else:
            self.ex_wiki_words = args.ex_wiki_words = ExWikiWords()

        if hasattr(args, 'w2v'):
            self.w2v = args.w2v
        else:
            self.w2v = args.w2v = W2V(args)

        if hasattr(args, 'w_freq'):
            self.w_freq = args.w_freq
        else:
            self.w_freq = args.w_freq = WFreq(args)

        if hasattr(args, 'ent_name_id'):
            self.ent_name_id = args.ent_name_id
        else:
            self.ent_name_id = args.ent_name_id = EntNameID(args)

        if hasattr(args, 'rewtr'):
            self.rewtr = args.rewtr
        else:
            self.rewtr = args.rewtr = REWTR(args)

        # **YD** main class "E2V" and "ModelA" and "BatchDatasetA"
        if hasattr(args, 'e2v'):
            self.e2v = args.e2v
        else:
            self.e2v = args.e2v = E2V(args)

        if hasattr(args, 'model'):
            self.model_a = args.model_a
        else:
            self.model_a = args.model_a = ModelA(args)

        if hasattr(args, 'batch_dataset_a'):
            self.batch_dataset_a = args.batch_dataset_a
        else:
            self.batch_dataset_a = args.batch_dataset_a = BatchDatasetA(args)


def train(args):

    if 'cuda' in args.type:
        assert hasattr(args, 'device')
        torch.cuda.set_device(args.device)

    learn_a = LearnA(args)

    # **YD** require model to be defined
    model = learn_a.model_a
    if 'cuda' in args.type:
        model.cuda()

    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # **YD** decide criterion (need to see manual to change)
    """
    if args.loss == 'neg' or args.loss == 'nce':
        criterion = nn.SoftMarginCriterion()

    elif args.loss == 'maxm':
        criterion = nn.MultiMarginCriterion(1, torch.ones(args.num_neg_words), 0.1)

    elif args.loss == 'is':
        criterion = nn.CrossEntropyCriterion()
    """

    # https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html#torch.nn.MultiMarginLoss
    # **YD**, is averaged by default, consistent with original implement
    # see test/criterion/* for more information
    if args.loss == 'maxm':
        criterion = nn.MultiMarginLoss(p=1, weight=torch.ones(args.num_neg_words), margin=0.1)
    else:
        raise ValueError('unknown criterion')

    if 'cuda' in args.type:
        criterion.cuda()

    #  **YD** decide optimizer (need to see manual to change)

    params = list(
        filter(
            lambda p: p.requires_grad,
            chain(model.parameters()),
        )
    )

    # **YD** for debugging
    # print(model)
    # print(model.parameters())

    if args.optimization == 'ADAGRAD':
        optimizer = torch.optim.Adagrad(params, lr=args.lr)
        """
        elif args.optimization == 'RMSPROP':
            optimizer = None
        elif args.optimization == 'SGD':
            optimizer = None
        elif args.optimization == 'ADAM':
            optimizer = None
        """
    else:
        raise ValueError('unknown optimizer')

    if 'cuda' in args.type:
        model.cpu()

    # --Show some entity examples
    # **YD** 'geom_unit_tests' not implemented
    learn_a.e2v.geom_unit_tests(model)

    if 'cuda' in args.type:
        model.cuda()

    # **YD** training status
    epoch = 0
    num_steps = 0
    num_instances = 0
    # **YD** not sure the usage
    model.train()
    # **YD** initial optimizer and parameters, like:
    # parameter = model.trainable_parameters()
    # optimizer = optimizer.Adam(lr=?, parameter=parameter)

    print('\n==> doing epoch on training data:')
    print("==> online epoch # " + str(epoch) + ' [batch size = ' + str(args.batch_size) + ']')

    # **YD** test stop criteria
    args.max_epoch = getattr(args, 'max_epoch', 500)
    args.max_num_steps = getattr(args, 'max_num_steps', math.inf)
    args.max_num_instances = getattr(args, 'max_num_instances', math.inf)

    while epoch < args.max_epoch and num_steps < args.max_num_steps and num_instances < args.max_num_instances:
        start_time = time.time()
        print('\n===> TRAINING EPOCH #' + str(epoch) + '; num batches ' + str(args.num_batches_per_epoch) + ' <===')

        # **YD** used to track loss before and after optimizer step
        avg_loss_before_opt_per_epoch = 0.0
        avg_loss_after_opt_per_epoch = 0.0

        for batch_index in tqdm(range(1, args.num_batches_per_epoch)):
            # -- Read one mini-batch from one data_thread:
            # **YD** get_minibatch has been implemented, may add extra input parameters
            inputs, targets = learn_a.batch_dataset_a.get_minibatch()

            # -- Move data to GPU:
            # **YD** minibatch_to_correct_type, correct_type not implemented
            # minibatch_to_correct_type(inputs)
            # targets = correct_type(targets)
            inputs = utils.move_to_cuda(inputs)
            targets = utils.move_to_cuda(targets)

            '''
            inputs = [
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
            '''

            optimizer.zero_grad()
            output = model(inputs)
            # print(output.size(), args.batch_size, args.num_words_per_ent, args.num_neg_words)
            assert output.size(0) == args.batch_size * args.num_words_per_ent
            assert output.size(1) == args.num_neg_words

            loss_before_opt = criterion(output, targets)
            avg_loss_before_opt_per_epoch += loss_before_opt
            loss_before_opt.backward()
            optimizer.step()

            '''
            loss_after_opt = criterion(model(inputs), targets)
            avg_loss_after_opt_per_epoch += loss_after_opt

            if loss_after_opt > loss_before_opt:
                print('!!!!!! LOSS INCREASED: ' + str(loss_before_opt) + ' --> ' + str(loss_after_opt))
            '''

            # **YD** customized to add stop criteria
            num_steps += 1
            num_instances += args.batch_size

            # **YD** break to debug test
            # break


        avg_loss_before_opt_per_epoch = avg_loss_before_opt_per_epoch / args.num_batches_per_epoch
        avg_loss_after_opt_per_epoch = avg_loss_after_opt_per_epoch / args.num_batches_per_epoch

        print('\nAvg loss before opt = ' + str(avg_loss_before_opt_per_epoch) +
              '; Avg loss after opt = ' + str(avg_loss_after_opt_per_epoch))

        use_time = (time.time() - start_time) / (args.num_batches_per_epoch * args.batch_size)
        print("this epoch takes: {}s".format(time.time() - start_time))
        print("==> time to learn 1 full entity = " + str(use_time * 1000) + 'ms')

        '''
        # --Show some entity examples
        # **YD** 'geom_unit_tests' has been implemented
        if 'cuda' in args.type:
            model.cpu()

        learn_a.e2v.geom_unit_tests(model)
        # -- Various testing measures:
        # **YD** 'compute_relatedness_metrics', 'entity_similarity' not implemented, evaluation on entity similarity
        if args.entities != '4EX' and epoch % args.test_every_num_epochs == 0:
            learn_a.rewtr.compute_relatedness_metrics(learn_a.e2v.entity_similarity, model)

        if 'cuda' in args.type:
            model.cuda()
        '''

        # -- Save model
        if epoch % args.save_every_num_epochs == 0:
            print('==> saving model: ' + args.root_data_dir + 'generated/ent_vecs/ent_vecs__ep_' + str(epoch) + '.t7')

            # **YD** rewrite not done, save entity embedding
            if not os.path.exists(os.path.join(args.root_data_dir, 'generated/ent_vecs/')):
                os.mkdir(os.path.join(args.root_data_dir, 'generated/ent_vecs/'))
            torch.save(
                F.normalize(model.entity_embedding.weight, 2),
                os.path.join(args.root_data_dir,'generated/ent_vecs/ent_vecs__ep_' + str(epoch) + '.pt'),
            )

        # -- next epoch
        epoch += 1


def main():

    parser = argparse.ArgumentParser(
        description='parameters to train entity embedding with word model',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--type',
        type=str,
        default='cuda',
        choices=['double', 'float', 'cuda'],
        help='Type: double | float | cuda',
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='Type: double | float | cuda',
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )

    parser.add_argument(
        '--optimization',
        type=str,
        default='ADAGRAD',
        choices=['RMSPROP', 'ADAGRAD', 'ADAM', 'SGD'],
        help='optimizer type',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.3,
        help='learning rate',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=500,
        help='Mini-batch size (1 = pure stochastic)',
    )

    parser.add_argument(
        '--word_vecs',
        type=str,
        default='w2v',
        choices=['glove', 'w2v'],
        help='300d word vectors type: glove | w2v',
    )

    parser.add_argument(
        '--num_words_per_ent',
        type=int,
        default=20,
        help='Num positive words sampled for the given entity at each iteration.',
    )

    parser.add_argument(
        '--num_neg_words',
        type=int,
        default=5,
        help='Num negative words sampled for each positive word.',
    )

    parser.add_argument(
        '--unig_power',
        type=float,
        default=0.6,
        help='Negative sampling unigram power (0.75 used in Word2Vec).',
    )

    parser.add_argument(
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', '4EX', 'ALL'],
        help='Set of entities for which we train embeddings: 4EX (tiny, for debug) |'
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    parser.add_argument(
        '--init_vecs_title_words',
        type=bool,
        default=True,
        #action='store_false',
        help='whether the entity embeddings should be initialized with the average of '
             'title word embeddings. Helps to speed up convergence speed of entity embeddings learning.',
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='maxm',
        choices=['nce', 'neg', 'is', 'maxm'],
        help='Loss function: nce (noise contrastive estimation) | '
             'neg (negative sampling) | is (importance sampling) | maxm (max-margin)',
    )

    parser.add_argument(
        '--data',
        type=str,
        default='wiki-canonical-hyperlinks',
        choices=['wiki-canonical', 'wiki-canonical-hyperlinks'],
        help='Training data: wiki-canonical (only) | wiki-canonical-hyperlinks',
    )

    parser.add_argument(
        '--num_passes_wiki_words',
        type=int,
        default=200,
        help='Num passes (per entity) over Wiki canonical pages before changing to using Wiki hyperlinks',
    )

    parser.add_argument(
        '--hyp_ctxt_len',
        type=int,
        default=10,
        help='Left and right context window length for hyperlinks.',
    )

    # add extra parameters to save and test
    parser.add_argument(
        '--test_every_num_epochs',
        type=int,
        default=1,
        help='number of epochs to do test',
    )

    parser.add_argument(
        '--save_every_num_epochs',
        type=int,
        default=3,
        help='number of epochs to save checkpoints',
    )

    parser.add_argument(
        '--word_vecs_size',
        type=int,
        default=300,
        help='dimension of word embedding',
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=300,
        help='max_num_epochs',
    )

    parser.add_argument(
        '--ent_vecs_size',
        type=int,
        default=300,
        help='dimension of entity embedding',
    )

    args = parser.parse_args()
    banner = ';obj-' + args.loss + ';' + args.data
    if args.data != 'wiki-canonical':
        banner += ';hypCtxtL-' + str(args.hyp_ctxt_len)
        banner += ';numWWpass-' + str(args.num_passes_wiki_words)

    banner += ';WperE-' + str(args.num_words_per_ent)
    banner += ';' + str(args.word_vecs) + ';negW-' + str(args.num_neg_words)
    banner += ';ents-' + str(args.entities) + ';unigP-' + str(args.unig_power)
    banner += ';bs-' + str(args.batch_size) + ';' + args.optimization + '-lr-' + str(args.lr)

    print('\n' + 'BANNER : ' + banner)
    print('\n===> RUN TYPE: ' + args.type)
    args.banner = banner

    if args.entities == 'ALL':
        args.num_batches_per_epoch = 4000
    elif args.entities == 'RLTD':
        args.num_batches_per_epoch = 2000
    elif args.entities == '4EX':
        args.num_batches_per_epoch = 400
    else:
        raise ValueError('unknown entity choices')

    assert hasattr(args, 'num_batches_per_epoch')

    train(args)


if __name__ == "__main__":
    main()