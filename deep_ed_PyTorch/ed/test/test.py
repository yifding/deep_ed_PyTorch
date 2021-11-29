import os
import sys
import time
import math

import torch

from deep_ed_PyTorch.ed.test.coref_persons import CorefPersons
from deep_ed_PyTorch.ed.test.ent_p_e_m_stats_test import new_ent_prior_map, add_prior_to_ent_prior_map, \
    print_ent_prior_maps_stats
from deep_ed_PyTorch.ed.test.ent_freq_stats_test import new_ent_freq_map, add_freq_to_ent_freq_map, \
    print_ent_freq_maps_stats

from deep_ed_PyTorch.ed.minibatch.build_minibatch import BuildMinibatch
from deep_ed_PyTorch.words.w_freq import WFreq
from deep_ed_PyTorch.entities.ent_name2id_freq import EntNameID, EFreqIndex


class Test(object):
    def __init__(self, args):
        self.args = args
        self.datasets = dict()
        for dataset in args.datasets:
            self.datasets[dataset] = os.path.join(
                args.root_data_dir,
                'generated/test_train_data/' + dataset + '.csv',
            )

        if hasattr(args, "build_minibatch"):
            self.build_minibatch = args.build_minibatch
        else:
            self.build_minibatch = args.build_minibatch = BuildMinibatch(args)

        if hasattr(args, 'ent_name_id'):
            self.ent_name_id = args.ent_name_id
        else:
            self.ent_name_id = args.ent_name_id = EntNameID(args)

        if hasattr(args, 'w_freq'):
            self.w_freq = args.w_freq
        else:
            self.w_freq = args.w_freq = WFreq(args)

        if hasattr(args, 'e_freq_index'):
            self.e_freq_index = args.e_freq_index
        else:
            self.e_freq_index = args.e_freq_index = EFreqIndex(args)

        if hasattr(args, 'coref_persons'):
            self.coref_persons = args.coref_persons
        else:
            self.coref_persons = args.coref_persons = CorefPersons(args)

    def test(self, epoch=0):
        # **YD** model may be required to insert in the function
        f1_scores = dict()
        for banner, dataset in self.datasets.items():
            print('start to test:', banner)
            self.test_one(banner, f1_scores, epoch)

        """ -- Plot accuracies
        if train_and_test then
        num_batch_train_between_plots = num_batches_per_epoch
        
        testAccLogger:add(f1_scores)
        styles = {}
        for banner,_ in pairs(f1_scores) do
          styles[banner] = '-'
        end
        testAccLogger:style(styles)
        testAccLogger:plot('F1', 'x ' .. num_batch_train_between_plots .. ' mini-batches')
        end
        """

    def test_one(self, banner, f1_scores, epoch):
        """
        test for one dataset
        :param banner: title(str) of a dataset
        :param f1_scores:  dictionary with banner as key
        :param epoch: number of epoch in the traininig procedure, also used to evaluate trained model

        :return: None
        """
        # -- Load dataset lines
        # **YD** "get_dataset_lines" has been implemented, return a dictionary of list
        dataset_lines = self.get_dataset_lines(banner)

        dataset_num_mentions = 0
        for doc_id, lines_map in dataset_lines.items():
            dataset_num_mentions += len(lines_map)

        print('\n===> ' + banner + '; num mentions = ' + '{}'.format(dataset_num_mentions))

        start_time = time.time()

        # **YD** "new_ent_freq_map", "new_ent_prior_map" has been implemented
        num_true_positives = 0.0
        grd_ent_freq_map = new_ent_freq_map()
        correct_classified_ent_freq_map = new_ent_freq_map()

        grd_ent_prior_map = new_ent_prior_map()
        correct_classified_ent_prior_map = new_ent_prior_map()

        num_mentions_without_gold_ent_in_candidates = 0
        both_pem_ours = 0 # -- num mentions solved both by argmax p(e|m) and our global model
        only_pem_not_ours = 0 # -- num mentions solved by argmax p(e|m), but not by our global model
        only_ours_not_pem = 0 # -- num mentions solved by our global model, but not by argmax p(e|m)
        not_ours_not_pem = 0 # -- num mentions not solved neither by our model nor by argmax p(e|m)

        processed_docs = 0
        processed_mentions = 0

        # -- NN forward pass:
        model = self.args.model
        model.eval()

        # go through each documents
        for doc_id, doc_lines in dataset_lines.items():
            processed_docs += 1
            num_mentions = len(doc_lines)
            processed_mentions += num_mentions

            # **YD** "empty_minibatch_with_ids" from build_minibatch has been implemented
            inputs = self.build_minibatch.empty_minibatch_with_ids(num_mentions)
            targets = torch.zeros(num_mentions, dtype=torch.long) * (-1)

            mentions = {}
            for k in range(num_mentions):
                sample_line = doc_lines[k]
                parts = sample_line.split('\t')
                mentions[k] = parts[2]

                # **YD** "process_one_line" from build_minibatch has been implemented
                target = self.build_minibatch.process_one_line(sample_line, inputs, k, False)
                targets[k] = target

            # **YD** "minibatch_to_correct_type" logic needs to rewrite
            inputs, targets = self.build_minibatch.minibatch_to_correct_type(inputs, targets, False)


            # **YD** local only? may rewrite logic
            # preds, beta = model(inputs)

            # **YD** "debug_softmax_word_weights" and "final_local_scores"
            # is prepared by the local model.

            # --------- Subnetworks used to print debug weights and scores :
            #     -- num_mentions x num_ctxt_vecs
            #     debug_softmax_word_weights = additional_local_submodels.model_debug_softmax_word_weights:forward(inputs):float()
            #     -- num_mentions, max_num_cand:
            #     final_local_scores = additional_local_submodels.model_final_local:forward(inputs):float()

            if self.args.type == 'cuda':
                model.cpu()
            preds, beta, entity_context_sim_scores = model(inputs)
            debug_softmax_word_weights = beta
            final_local_scores = entity_context_sim_scores

            # -- Process results:
            # **YD** "get_cand_ent_wikiids" has been implemented
            # print('inputs[1][0]', type(inputs[1][0]), inputs[1][0])
            all_ent_wikiids = self.build_minibatch.get_cand_ent_wikiids(inputs)
            # print('all_ent_wikiids', type(all_ent_wikiids), all_ent_wikiids)

            for k in range(num_mentions):
                pred = preds[k]
                assert not torch.isnan(torch.norm(pred))

                # -- Ignore unk entities (padding entities):
                # **YD** "all_ent_wikiids" has been implemented
                ent_wikiids = all_ent_wikiids[k]

                # **YD** "max_num_cand", "unk_ent_wikiid" has been implemented
                for i in range(self.args.max_num_cand):
                    if ent_wikiids[i] == self.ent_name_id.unk_ent_wikiid:
                        pred[i] = -1e8  # --> -infinity

                # -- PRINT DEBUG SCORES: Show network weights and scores for entities with a valid gold label.
                if targets[k] >= 0:
                    # **YD** "get_log_p_e_m" has been implemented
                    log_p_e_m = self.build_minibatch.get_log_p_e_m(inputs)

                    if k == 0:
                        print('\n')
                        print('============================================')
                        print('============ DOC : ' + '{}'.format(doc_id) + ' ================')
                        print('============================================')

                    # -- Winning entity
                    _, argmax_idx = torch.max(pred, 0)
                    win_idx = argmax_idx.item()  # -- the actual number
                    ent_win = ent_wikiids[win_idx].item()

                    # **YD** "get_ent_name_from_wikiid" has been implemented
                    ent_win_name = self.ent_name_id.get_ent_name_from_wikiid(ent_win)
                    ent_win_log_p_e_m = log_p_e_m[k][win_idx]

                    # **YD** "final_local_scores" has been implemented
                    ent_win_local = final_local_scores[k][win_idx]

                    # -- Just some sanity check
                    best_pred, best_pred_idxs = torch.topk(pred, 1)
                    if pred[best_pred_idxs] != best_pred:
                        print(pred)

                    assert (pred[best_pred_idxs] == best_pred)
                    assert (pred[best_pred_idxs] == pred[win_idx])

                    # -- Grd trth entity
                    grd_idx = targets[k].item()
                    ent_grd = ent_wikiids[grd_idx].item()
                    # **YD** "get_ent_name_from_wikiid" has been implemented
                    ent_grd_name = self.ent_name_id.get_ent_name_from_wikiid(ent_grd)
                    ent_grd_log_p_e_m = log_p_e_m[k][grd_idx]

                    # **YD** "final_local_scores" has been implemented
                    ent_grd_local = final_local_scores[k][grd_idx]

                    # print('ent_wikiids', type(ent_wikiids), ent_wikiids)
                    # print('win_idx', type(win_idx), win_idx)
                    # print('grd_idx', type(grd_idx), grd_idx)
                    if win_idx != grd_idx:
                        assert ent_win != ent_grd
                        print(
                            '\n====> ' + 'INCORRECT ANNOTATION' +
                            ' : mention = ' + mentions[k] +
                            ' ==> ENTITIES (OURS/GOLD): ' + ent_win_name +
                            ' <---> ' + ent_grd_name
                        )

                    else:
                        assert ent_win == ent_grd
                        mention_str = 'mention = ' + mentions[k]
                        if math.exp(ent_grd_log_p_e_m) >= 0.99:
                            mention_str = 'High Prob:' + mention_str

                        print(
                            '\n====> ' + 'CORRECT ANNOTATION' +
                            ' : ' + mention_str +
                            ' ==> ENTITY: ' + ent_grd_name
                        )

                    print(
                        'SCORES: global= ', float(pred[win_idx]), float(pred[grd_idx]),
                        '; local(<e,ctxt>)= ', float(ent_win_local), float(ent_grd_local),
                        '; log p(e|m)= ', float(ent_win_log_p_e_m), float(ent_grd_log_p_e_m)
                    )

                    # -- Print top attended ctxt words and their attention weights:
                    str_words = '\nTop context words (sorted by attention weight, ' \
                                'only non-zero weights - top R words): \n'

                    # **YD** "get_ctxt_word_ids" has been implemented
                    ctxt_word_ids = self.build_minibatch.get_ctxt_word_ids(inputs)
                    # -- num_mentions x opt.ctxt_window

                    # **YD** "debug_softmax_word_weights" logic has been implemented
                    best_scores, best_word_idxs = torch.topk(debug_softmax_word_weights[k], self.args.ctxt_window)

                    seen_unk_w_id = False
                    for wk in range(self.args.ctxt_window):
                        w_idx_ctxt = best_word_idxs[wk]
                        assert 0 <= w_idx_ctxt < self.args.ctxt_window
                        w_id = ctxt_word_ids[k][w_idx_ctxt]
                        # **YD** "unk_w_id" has been implemented
                        if w_id != self.w_freq.unk_w_id or not seen_unk_w_id:
                            if w_id == self.w_freq.unk_w_id:
                                seen_unk_w_id = True
                            # **YD** "get_word_from_id" has been implemented
                            w = self.w_freq.get_word_from_id(w_id)
                            score = debug_softmax_word_weights[k][w_idx_ctxt]
                            assert score == best_scores[wk]

                            if score > 0.001:
                                str_words += w + '[' + '{:.3f}'.format(score) + ']; '

                    print(str_words)
                # ----------------- Done printing scores and weights

                # -- Count how many of the winning entities do not have a valid ent vector
                _, argmax_idx = torch.max(pred, 0)

                # **YD** "get_thid", "unk_ent_thid" has been implemented
                if targets[k] >= 0 and \
                        self.ent_name_id.get_thid(ent_wikiids[argmax_idx.item()].item()) == self.ent_name_id.unk_ent_thid:
                    print(pred)
                    print(ent_wikiids)
                    print('\n\n', '!!!!Entity w/o vec: ', ent_wikiids[argmax_idx.item()], ' line = ', doc_lines[k])
                    sys.exit()

                # -- Accumulate statistics about the annotations of our system.
                if targets[k] >= 0:
                    # **YD** "get_log_p_e_m" has been implemented
                    log_p_e_m = self.build_minibatch.get_log_p_e_m(inputs)
                    nn_is_good = True
                    pem_is_good = True

                    # -- Grd trth entity
                    grd_idx = targets[k]

                    for j in range(self.args.max_num_cand):
                        if j != grd_idx and pred[grd_idx] < pred[j]:
                            # **YD** "unk_ent_wikiid" has been implemented
                            assert ent_wikiids[j] != self.ent_name_id.unk_ent_wikiid
                            nn_is_good = False

                        if j != grd_idx and log_p_e_m[k][grd_idx] < log_p_e_m[k][j]:
                            # **YD** "unk_ent_wikiid" has been implemented
                            assert ent_wikiids[j] != self.ent_name_id.unk_ent_wikiid
                            pem_is_good = False

                    if nn_is_good and pem_is_good:
                        both_pem_ours += 1
                    elif nn_is_good:
                        only_ours_not_pem += 1
                    elif pem_is_good:
                        only_pem_not_ours += 1
                    else:
                        not_ours_not_pem += 1

                    # **YD** "unk_ent_wikiid" has been implemented
                    assert ent_wikiids[targets[k]] != self.ent_name_id.unk_ent_wikiid, ' Something is terribly wrong here '

                    # confusion:add(pred, targets[k])

                    # -- Update number of true positives
                    _, argmax_idx = torch.max(pred, 0)
                    winning_entiid = ent_wikiids[argmax_idx.item()].item()
                    grd_entiid = ent_wikiids[targets[k]].item()

                    # **YD** "get_ent_freq", "add_freq_to_ent_freq_map",
                    # "add_prior_to_ent_prior_map" has been implemented
                    grd_ent_freq = self.e_freq_index.get_ent_freq(grd_entiid)
                    add_freq_to_ent_freq_map(grd_ent_freq_map, grd_ent_freq)

                    grd_ent_prior = math.exp(log_p_e_m[k][grd_idx])
                    add_prior_to_ent_prior_map(grd_ent_prior_map, grd_ent_prior)

                    if winning_entiid == grd_entiid:
                        add_freq_to_ent_freq_map(correct_classified_ent_freq_map, grd_ent_freq)
                        add_prior_to_ent_prior_map(correct_classified_ent_prior_map, grd_ent_prior)
                        num_true_positives += 1

                else:   # -- grd trth is not between the given set of candidates, so we cannot be right
                    num_mentions_without_gold_ent_in_candidates += 1

                    # confusion:add(torch.zeros(max_num_cand), max_num_cand)

                # -- disp progress
                # xlua.progress(processed_mentions, dataset_num_mentions)

        # -- done with this mini batch

        # -- Now plotting results
        use_time = time.time() - start_time
        time_per_num_mentions = use_time / dataset_num_mentions
        print("==> time to test 1 sample = " + "{0:.3f}".format(time_per_num_mentions * 1000) + "ms")

        # confusion:__tostring__()

        # -- We refrain from solving mentions without at least one candidate
        # **YD** "get_dataset_num_non_empty_candidates" has been implemented
        precision = 100.0 * num_true_positives / self.get_dataset_num_non_empty_candidates(dataset_lines)
        recall = 100.0 * num_true_positives / dataset_num_mentions

        # assert(math.abs(recall - confusion.totalValid * 100.0) < 0.01, 'Difference in recalls.')
        f1 = 2.0 * precision * recall / (precision + recall)
        f1_scores[banner] = f1

        f1_str = "{0:.2f}".format(f1) + '%'

        '''
        if banner == 'aida-A' and f1 >= 90.20 then
        f1_str = green(string.format("%.2f", f1) .. '%')
        '''

        print(
            '==> ' + banner + ' ' + banner + ' ; EPOCH = ' + str(epoch) +
            ': Micro recall = ' + '{0:.2f}'.format(recall * 100.0) + '%' +
            ' ; Micro F1 = ' + f1_str
        )


        '''
        -- Lower learning rate if we got close to minimum
        if banner == 'aida-A' and f1 >= 90 then
        opt.lr = 1e-5
        end
        '''

        # -- We slow down training a little bit if we passed the 90% F1 score threshold.
        # -- And we start saving (good quality) models from now on.

        '''
        if banner == 'aida-A' then
            if f1 >= 80.0 then
              opt.save = true
            else
              opt.save = false
            end
          end
        '''

        # "print_ent_freq_maps_stats", "print_ent_prior_maps_stats" has been implemented
        print_ent_freq_maps_stats(correct_classified_ent_freq_map, grd_ent_freq_map)
        print_ent_prior_maps_stats(correct_classified_ent_prior_map, grd_ent_prior_map)

        print(
              ' num_mentions_w/o_gold_ent_in_candidates = ',
              num_mentions_without_gold_ent_in_candidates,
              ' total num mentions in dataset = ',
              dataset_num_mentions,
        )

        print(
                ' percentage_mentions_w/o_gold_ent_in_candidates = ' +
                '{0:.2f}'.format(100.0 * num_mentions_without_gold_ent_in_candidates / dataset_num_mentions) +
                '%;  percentage_mentions_solved : both_pem_ours = ' +
                '{0:.2f}'.format(100.0 * both_pem_ours / dataset_num_mentions) +
                '%;  only_pem_not_ours = ' +
                '{0:.2f}'.format(100.0 * only_pem_not_ours / dataset_num_mentions) +
                '%;  only_ours_not_pem = ' +
                '{0:.2f}'.format(100.0 * only_ours_not_pem / dataset_num_mentions) +
                '%;  not_ours_not_pem = ' +
                '{0:.2f}'.format(100.0 * not_ours_not_pem / dataset_num_mentions) + '%'
        )

    def get_dataset_lines(self, banner):
        all_doc_lines = dict()
        with open(self.datasets[banner]) as reader:
            for line in reader:
                line = line.rstrip()
                parts = line.split('\t')
                doc_name = parts[0]
                if doc_name not in all_doc_lines:
                    # **YD** interface may be problematic
                    all_doc_lines[doc_name] = list()
                all_doc_lines[doc_name].append(line)
        # -- Gather coreferent mentions to increase accuracy.
        # **YD** "build_coreference_dataset" not implemented
        return self.coref_persons.build_coreference_dataset(all_doc_lines, banner)

    def get_dataset_num_non_empty_candidates(self, dataset_lines):
        num_predicted = 0
        # **YD** interface may be problematic
        for doc_id, lines_map in dataset_lines.items():
            # **YD** interface may be problematic
            for sample_line in lines_map:
                parts = sample_line.split('\t')
                if parts[6] != 'EMPTYCAND':
                    num_predicted += 1
        return num_predicted

















