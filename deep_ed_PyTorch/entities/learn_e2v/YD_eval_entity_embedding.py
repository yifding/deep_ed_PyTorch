import os
import argparse

import torch

from learn_a import LearnA


def eva(args):
    learn_a = LearnA(args)

    # load model, and reset its entity embedding from trained file
    model = learn_a.model_a

    dir = args.dir

    for file in os.listdir(dir):
        pt_file = os.path.join(dir, file)
        print('*' * 100)
        print(file)
        weight = torch.load(pt_file, map_location=torch.device('cpu'))
        model.entity_embedding.weight.data.copy_(weight)
        model.eval()

        learn_a.e2v.geom_unit_tests(model)
        learn_a.rewtr.compute_relatedness_metrics(learn_a.e2v.entity_similarity, model)

        print('-' * 100)


def main():

    parser = argparse.ArgumentParser(
        description='parameters to train entity embedding with word model',
        allow_abbrev=False,
    )



    parser.add_argument(
        '--dir',
        type=str,
        default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/generated/ent_vecs',
        help='data dir to store entity embedding',
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
        default='/home/yding4/EL_resource/data/deep_ed_PyTorch_data/',
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
        '--ent_vecs_size',
        type=int,
        default=300,
        help='dimension of entity embedding',
    )

    args = parser.parse_args()
    args.init_vecs_title_words = False
    eva(args)


if __name__ == '__main__':
    main()


