# -- The code in this file does two things:
# --   a) extracts and puts the entity relatedness dataset in two maps (reltd_validate and
# --      reltd_test). Provides functions to evaluate entity embeddings on this dataset
# --      (Table 1 in our paper).
# --   b) extracts all entities that appear in any of the ED (as mention candidates) or
# --      entity relatedness datasets. These are placed in an object called rewtr that will
# --      be used to restrict the set of entities for which we want to train entity embeddings
# --      (done with the file entities/learn_e2v/learn_a.lua).

import os
import torch
import math
import argparse

from deep_ed_PyTorch.entities.EX_wiki_words import ExWikiWords


class REWTR(object):
    def __init__(self, args):
        self.args = args

        self.rel_validate_txt = 'basic_data/relatedness/validate.svm'
        self.rel_validate_txtfilename = os.path.join(args.root_data_dir, self.rel_validate_txt)
        self.rel_validate_torch = 'generated/relatedness_validate.pt'
        self.rel_validate_torch_file = os.path.join(args.root_data_dir, self.rel_validate_torch)

        self.rel_test_txt = 'basic_data/relatedness/test.svm'
        self.rel_test_txtfilename = os.path.join(args.root_data_dir, self.rel_test_txt)
        self.rel_test_torch = 'generated/relatedness_test.pt'
        self.rel_test_torch_file = os.path.join(args.root_data_dir, self.rel_test_torch)

        self.ent_lines_4EX = ExWikiWords()

        # **YD** self.load_reltd_set has been implemented
        """
        reltd structure (all number is in string format used in torch):
            {
                'q' (query number, keys) : {
                'e1' (single key): e1 ,
                'cand' (single key): {
                    e2 (e2 number, keys): related or not (0 or 1),
                    },
                } 
            }
        """
        self.reltd_validate = self.load_reltd_set(self.rel_validate_torch_file,
                                                  self.rel_validate_txtfilename,
                                                  'validate')

        self.reltd_test = self.load_reltd_set(self.rel_test_torch_file,
                                              self.rel_test_txtfilename,
                                              'test')

        # **YD** self.extract_reltd_ents has been implemented
        self.reltd_ents_direct_validate = self.extract_reltd_ents(self.reltd_validate)
        self.reltd_ents_direct_test = self.extract_reltd_ents(self.reltd_test)

        self.rewtr_torch = 'generated/all_candidate_ents_ed_rltd_datasets_RLTD.pt'
        self.rewtr_torch_file = os.path.join(args.root_data_dir, self.rewtr_torch)

        print('==> Loading relatedness thid tensor')
        if not os.path.isfile(self.rewtr_torch_file):
            print('  ---> torch file NOT found. Loading reltd_ents_wikiid_to_rltdid from txt file instead (slower).')
            # -- Gather the restricted set of entities for which we train entity embeddings:
            rltd_all_ent_wikiids = dict()

            # -- 1) From the relatedness dataset
            for ent_wikiid in self.reltd_ents_direct_validate:
                rltd_all_ent_wikiids[ent_wikiid] = 1

            for ent_wikiid in self.reltd_ents_direct_test:
                rltd_all_ent_wikiids[ent_wikiid] = 1

            # print('after validate/test dataset: ', len(rltd_all_ent_wikiids))

            # -- 1.1) From a small dataset (used for debugging / unit testing).
            # **YD** ent_lines_4EX has been implemented
            for line in self.ent_lines_4EX.ent_lines_4EX.values():
                line = line.strip('\t\n')
                parts = line.split('\t')
                assert len(parts) == 3

                ent_wikiid = int(parts[0])
                assert ent_wikiid > 0, 'invalid entity wikiid'
                rltd_all_ent_wikiids[ent_wikiid] = 1

            # print('after small dataset: ', len(rltd_all_ent_wikiids))
            # -- 2) From all ED datasets:
            files = ['aida_train.csv', 'aida_testA.csv', 'aida_testB.csv',
                     'wned-aquaint.csv', 'wned-msnbc.csv', 'wned-ace2004.csv',
                     'wned-clueweb.csv', 'wned-wikipedia.csv']

            for file in files:
                file_file = os.path.join(args.root_data_dir, 'generated/test_train_data/' + file)
                with open(file_file, 'r') as reader:
                    for line in reader:
                        line = line.rstrip('\t\n')
                        parts = line.split('\t')

                        assert parts[5] == 'CANDIDATES'
                        assert parts[-2] == 'GT:'
                        if parts[6] != 'EMPTYCAND':
                            # **YD** huge bug! for part in parts[7:-2]:
                            for part in parts[6:-2]:
                                p = part.split(',')
                                ent_wikiid = int(p[0])
                                rltd_all_ent_wikiids[ent_wikiid] = 1

                            p = parts[-1].split(',')
                            if len(p) >= 2:
                                ent_wikiid = int(p[1])
                                assert ent_wikiid > 0, 'invalid wikiid'

            # print('after EL dataset: ', len(rltd_all_ent_wikiids))

            # -- Insert unk_ent_wikiid
            # self.unk_ent_wikiid = 1
            unk_ent_wikiid = 1
            rltd_all_ent_wikiids[unk_ent_wikiid] = 1

            for wikiid in rltd_all_ent_wikiids:
                if type(wikiid) is not int:
                    raise ValueError('invalid type {}'.format(type(wikiid)))

            sorted_rltd_all_ent_wikiids = sorted(wikiid for wikiid in rltd_all_ent_wikiids)
            reltd_ents_wikiid_to_rltdid = dict((wikiid, index)
                                               for index, wikiid in enumerate(sorted_rltd_all_ent_wikiids))

            reltd_ents_rltdid_to_wikiid = dict((index, wikiid)
                                               for wikiid, index in reltd_ents_wikiid_to_rltdid.items())

            self.rewtr = dict()
            self.rewtr['reltd_ents_wikiid_to_rltdid'] = reltd_ents_wikiid_to_rltdid
            self.rewtr['reltd_ents_rltdid_to_wikiid'] = reltd_ents_rltdid_to_wikiid
            self.rewtr['num_rltd_ents'] = len(reltd_ents_rltdid_to_wikiid)

            print('Writing reltd_ents_wikiid_to_rltdid to pt File for future usage.')
            torch.save(self.rewtr, self.rewtr_torch_file)

        else:
            self.rewtr = torch.load(self.rewtr_torch_file)

        print('    Done loading relatedness sets. Num queries test = ', len(self.reltd_test),
              '. Num queries valid = ', len(self.reltd_validate),'. Total num ents restricted set = ',
              self.rewtr['num_rltd_ents'])

    @property
    def reltd_ents_wikiid_to_rltdid(self):
        return self.rewtr['reltd_ents_wikiid_to_rltdid']

    @property
    def reltd_ents_rltdid_to_wikiid(self):
        return self.rewtr['reltd_ents_rltdid_to_wikiid']

    @property
    def num_rltd_ents(self):
        return self.rewtr['num_rltd_ents']

    def load_reltd_set(self, torch_file, txt_file, set_type='validate'):
        print('==> Loading relatedness ' + set_type)
        if not os.path.isfile(torch_file):
            print('  ---> torch file NOT found. Loading relatedness ' + set_type + ' from txt file instead (slower).')
            reltd = dict()
            with open(txt_file, 'r') as reader:
                for line in reader:
                    line = line.rstrip('\t\n')
                    parts = line.split(' ')
                    label = int(parts[0])
                    assert label == 0 or label == 1
                    t = parts[1].split(':')
                    q = int(t[1])
                    i = 1
                    while parts[i] != '#':
                        i += 1
                    i += 1

                    ents = parts[i].split('-')
                    e1 = int(ents[0])
                    e2 = int(ents[1])

                    if q not in reltd:
                        reltd[q] = dict()
                        reltd[q]['e1'] = e1
                        reltd[q]['cand'] = dict()

                    reltd[q]['cand'][e2] = label

            print('    Done loading relatedness ' , set_type, '. Num queries = ', len(reltd))
            print('Writing pt File for future usage. Next time relatedness dataset will load faster!')
            torch.save(reltd, torch_file)
            print('    Done saving.')
            return reltd
        else:
            reltd = torch.load(torch_file)
            return reltd

    def extract_reltd_ents(self, reltd_set):
        reltd_ents_direct = dict()
        for v in reltd_set.values():
            reltd_ents_direct[v['e1']] = 1
            for e2 in v['cand']:
                reltd_ents_direct[e2] = 1
        return reltd_ents_direct

    def compute_relatedness_metrics(self, entity_sim, model_a):
        self.compute_relatedness_metrics_from_maps(entity_sim, self.reltd_validate, self.reltd_test, model_a)

    def compute_relatedness_metrics_from_maps(self, entity_sim, validate_set, test_set, model_a):
        print('Entity Relatedness quality measure:')

        # **YD** compute_ideal_rltd_scores has been implemented
        ideals_rltd_validate_scores = self.compute_ideal_rltd_scores(validate_set)
        ideals_rltd_test_scores = self.compute_ideal_rltd_scores(test_set)

        # **YD** compute_MAP has been implemented
        assert abs(-1 + self.compute_MAP(ideals_rltd_validate_scores, validate_set)) < 0.001
        assert abs(-1 + self.compute_MAP(ideals_rltd_test_scores, test_set)) < 0.001

        # **YD** compute_e2v_rltd_scores has been implemented
        scores_validate = self.compute_e2v_rltd_scores(validate_set, entity_sim, model_a)
        scores_test = self.compute_e2v_rltd_scores(test_set, entity_sim, model_a)

        validate_table = dict()
        validate_table['scores'] = scores_validate
        validate_table['ideals_rltd_scores'] = ideals_rltd_validate_scores
        validate_table['reltd'] = validate_set

        test_table = dict()
        test_table['scores'] = scores_test
        test_table['ideals_rltd_scores'] = ideals_rltd_test_scores
        test_table['reltd'] = test_set

        # **YD** compute_MAP has been implemented
        # **YD** compute_NDCG has been implemented
        map_validate = self.compute_MAP(scores_validate, validate_set)
        ndcg_1_validate = self.compute_NDCG(1, validate_table)
        ndcg_5_validate = self.compute_NDCG(5, validate_table)
        ndcg_10_validate = self.compute_NDCG(10, validate_table)

        total = map_validate + ndcg_1_validate + ndcg_5_validate + ndcg_10_validate

        # **YD** blue_num_str not implemented
        map_validate_str = map_validate
        ndcg_1_validate_str = ndcg_1_validate
        ndcg_5_validate_str = ndcg_5_validate
        ndcg_10_validate_str = ndcg_10_validate
        total_str = total

        map_test = '{0:.3f}'.format(self.compute_MAP(scores_test, test_set))
        ndcg_1_test = '{0:.3f}'.format(self.compute_NDCG(1, test_table))
        ndcg_5_test = '{0:.3f}'.format(self.compute_NDCG(5, test_table))
        ndcg_10_test = '{0:.3f}'.format(self.compute_NDCG(10, test_table))

        print('measure    =', 'NDCG1', 'NDCG5', 'NDCG10', 'MAP', 'TOTAL VALIDATION')
        print('our (vald) =', ndcg_1_validate_str, ndcg_5_validate_str,
              ndcg_10_validate_str, map_validate_str, total_str)
        print('our (test) =', ndcg_1_test, ndcg_5_test, ndcg_10_test, map_test)
        print('Yamada\'16  =', 0.59, 0.56, 0.59, 0.52)
        print('WikiMW      =', 0.54, 0.52, 0.55, 0.48)

    def compute_NDCG(self, k, all_table):
        sum_ndcg = 0.0
        num_queries = 0
        for q in all_table['scores']:
            dcg = self.compute_DCG(k, q, all_table['scores'][q], all_table['reltd'])
            idcg = self.compute_DCG(k, q, all_table['ideals_rltd_scores'][q], all_table['reltd'])
            assert dcg <= idcg, str(dcg) + ' ' + str(idcg)
            sum_ndcg += dcg / idcg
            num_queries = num_queries + 1

        assert num_queries == len(all_table['reltd'])
        return sum_ndcg / num_queries

    """
    **YD** the original format of dcg is not right.
    original code at:
     https://github.com/dalab/deep-ed/blob/master/entities/relatedness/relatedness.lua#L146
    -- NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    local function compute_DCG(k, q, scores_q, reltd)
      local dcg = 0.0
      local i = 0
      for _,c in pairs(scores_q) do
        local label = reltd[q].cand[c.e2]
        i = i + 1
        if (label == 1) and i <= k then
          dcg = dcg + (1.0 / math.log(math.max(2,i) + 0.0, 2))
        end
      end
      return dcg
    end
    """

    # -- NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG
    def compute_DCG(self, k, q, scores_q, reltd):
        dcg = 0.0
        i = 0
        for c in scores_q:
            label = int(reltd[q]['cand'][c['e2']])
            i += 1
            if label == 1 and i <= k:
                dcg += 1.0 / math.log2(i + 1)

        return dcg

    # -- Mean Average Precision:
    # -- https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision
    # https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision
    def compute_MAP(self, scores, reltd):
        sum_avgp = 0.0
        num_queries = 0
        for q in scores:
            avgp = 0.0
            num_rel_ents_so_far = 0
            num_ents_so_far = 0.0
            for c in scores[q]:
                e2 = c['e2']
                label = reltd[q]['cand'][e2]
                num_ents_so_far += 1.0
                if int(label) == 1:
                    num_rel_ents_so_far += 1
                    precision = num_rel_ents_so_far / num_ents_so_far
                    avgp += precision

            avgp = avgp / num_rel_ents_so_far
            sum_avgp += avgp
            num_queries = num_queries + 1

        assert num_queries == len(reltd)
        return sum_avgp/num_queries

    # -- computes rltd scores based on ground truth labels
    def compute_ideal_rltd_scores(self, reltd):
        scores = dict()
        for q in reltd:
            scores[q] = []
            for e2, label in reltd[q]['cand'].items():
                aux = dict()
                aux['e2'] = e2
                aux['score'] = int(label)
                scores[q].append(aux)
            scores[q] = sorted(scores[q], key=lambda x: x['score'], reverse=True)
        return scores

    # -- computes rltd scores based on a given entity_sim function
    def compute_e2v_rltd_scores(self, reltd, entity_sim, model_a):
        # **YD** using number string instead of number may cause trouble in embedding dict usage
        scores = dict()
        for q in reltd:
            scores[q] = []
            for e2 in reltd[q]['cand']:
                aux = dict()
                aux['e2'] = e2

                # **YD** debug
                test_score = entity_sim(reltd[q]['e1'], e2, model_a)
                # print(test_score.dtype)
                # print(test_score.shape)
                # print(test_score)

                aux['score'] = float(entity_sim(reltd[q]['e1'], e2, model_a))
                scores[q].append(aux)
                scores[q] = sorted(scores[q], key=lambda x: x['score'], reverse=True)
        return scores


def test(args):
    REWTR(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate wiki hyperlink words rltd'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        # default='/scratch365/yding4/deep_ed_PyTorch/data/',
        required=True,
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    print(args)
    test(args)