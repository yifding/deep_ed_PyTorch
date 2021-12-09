import argparse


def arg_parse():

    parser = argparse.ArgumentParser(
        description='parser for ed model',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    parser.add_argument(
        '--type',
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'],
        help='Type: cpu | cuda',
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='index of GPU',
    )

    parser.add_argument(
        '--store_train_data',
        type=str,
        default='RAM',
        choices=['RAM', 'DISK'],
        help='Where to read the training data from, RAM to put training instances in RAM, which has enought space'
             'to store aida-train dataset',
    )

    parser.add_argument(
        '--optimization',
        type=str,
        default='ADAM',
        choices=['ADADELTA', 'ADAGRAD', 'ADAM', 'SGD'],
        help='optimizer type',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate',
    )

    parser.add_argument(
        '--batch_size', '-bs',
        type=int,
        default=1,
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
        '--entities',
        type=str,
        default='RLTD',
        choices=['RLTD', 'ALL'],
        help='Set of entities for which we train embeddings: '
             ' RLTD (restricted set) | ALL (all Wiki entities, too big to fit on a single GPU)',
    )

    parser.add_argument(
        '--ent_vecs_filename',
        type=str,
        # default='ent_vecs__ep_231.pt',
        required=True,
        help='File name containing entity vectors generated with entities/learn_e2v/learn_a.py',
    )

    parser.add_argument(
        '--ctxt_window', '-ctxtW',
        type=int,
        default=100,
        help='Number of context words at the left plus right of each mention',
    )

    parser.add_argument(
        '--top_ctxt_words', '-R',
        type=int,
        default=25,
        help='Hard attention threshold: top R context words are kept, the rest are discarded',
    )

    parser.add_argument(
        '--model_type',
        type=str,
        default='local',
        choices=['local', 'global'],
        help='model types to perform entity disambiguration',
    )

    parser.add_argument(
        '--nn_pem_interm_size',
        type=int,
        default=100,
        help='Number of hidden units in the f function described in Section 4 - Local score combination',
    )

    parser.add_argument(
        '--mat_reg_norm',
        type=int,
        default=1,
        help='Maximum norm of columns of matrices of the f network.',
    )

    parser.add_argument(
        '--lbp_iter', '-T',
        type=int,
        default=10,
        help='Number iterations of LBP hard-coded in a NN. Referred as T in the paper.',
    )

    parser.add_argument(
        '--lbp_damp',
        type=float,
        default=0.5,
        help='Damping factor for LBP.',
    )

    parser.add_argument(
        '--num_cand_before_rerank',
        type=int,
        default=30,
        help='number of candidates before rerank.',
    )

    parser.add_argument(
        '--keep_p_e_m',
        type=int,
        default=4,
        help='number of prior entities kept.',
    )

    parser.add_argument(
        '--keep_e_ctxt',
        type=int,
        default=3,
        help='number of context entites kept.',
    )

    parser.add_argument(
        '--loss',
        type=str,
        default='maxm',
        choices=['maxm', 'nll'],
        help='maxm (max-margin) or nll loss',
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
        '--R',
        type=int,
        default=25,
        help='hard threshold',
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

    parser.add_argument(
        '--save_interval',
        type=int,
        default=1,
        help='epochs interval to save model',
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=400,
        help='max number of epochs',
    )

    # -- Each batch is one document, so we test/validate/save the current model after each set of
    # -- 5000 documents. Since aida-train contains 946 documents, this is equivalent with 5 full epochs.
    parser.add_argument(
        '--num_batches_per_epoch',
        type=int,
        default=5000,
        help='number of batches in a epoch',
    )

    # **YD** ""w_freq_index"" builder #
    parser.add_argument(
        '--unig_power',
        type=float,
        default=0.6,
        help='Negative sampling unigram power (0.75 used in Word2Vec).',
    )

    # **YD** datasets for evaluation
    parser.add_argument(
        '--datasets',
        type=eval,
        default="['aida_testA', "
                "'aida_testB', "
                "'wned-msnbc', "
                "'wned-aquaint', "
                "'wned-ace2004', "
                "'aida_train', "
                "'wned-clueweb', "
                "'wned-wikipedia']",
        help='datasets for evaluation',
    )

    parser.add_argument(
        '--test_one_model_file',
        type=str,
        default="125.pt",
        help='model for evaluation',
    )


    args = parser.parse_args()

    # -- Whether to save the current ED model or not during training.
    # -- It will become true after the model gets > 90% F1 score on validation set (see test.lua).
    # **YD** should release the constraints to save every epoch

    args.save = True

    # consider persons' name coreference
    args.coref = True
    args.max_num_cand = args.keep_p_e_m + args.keep_e_ctxt

    return args

#  **YD** other support/util functions seems like to be not important.
