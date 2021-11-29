# -- Given a dataset, try to retrieve better entity candidates
# -- for ambiguous mentions of persons. For example, suppose a document
# -- contains a mention of a person called 'Peter Such' that can be easily solved with
# -- the current system. Now suppose that, in the same document, there
# -- exists a mention 'Such' referring to the same person. For this
# -- second highly ambiguous mention, retrieving the correct entity in
# -- top K candidates would be very hard. We adopt here a simple heuristical strategy of
# -- searching in the same document all potentially coreferent mentions that strictly contain
# -- the given mention as a substring. If such mentions exist and they refer to
# -- persons (contain at least one person candidate entity), then the ambiguous
# -- shorter mention gets as candidates the candidates of the longer mention.

import os

from deep_ed_PyTorch.entities.ent_name2id_freq.ent_name_id import EntNameID


class CorefPersons(object):
    def __init__(self, args):
        self.args = args
        if hasattr(args, 'ent_name_id'):
            self.ent_name_id = args.ent_name_id
        else:
            self.ent_name_id = args.ent_name_id = EntNameID(args)

        self.person_file = os.path.join(self.args.root_data_dir, 'basic_data/p_e_m_data/persons.txt')
        self.persons_ent_wikiids = self.load()

    def load(self):
        persons_ent_wikiids = dict()
        with open(self.person_file, 'r') as reader:
            for line in reader:
                line = line.rstrip()
                # **YD** "get_ent_wikiid_from_name", "unk_ent_wikiid" has been implemented
                ent_wikiid = self.ent_name_id.get_ent_wikiid_from_name(line, True)
                if ent_wikiid != self.ent_name_id.unk_ent_wikiid:
                    persons_ent_wikiids[ent_wikiid] = 1
        print('    Done loading persons index. Size = ',  len(persons_ent_wikiids))
        return persons_ent_wikiids

    def is_person(self, ent_wikiid):
        return ent_wikiid in self.persons_ent_wikiids

    def mention_refers_to_person(self, m, mention_ent_cand):
        top_p = 0
        top_ent = -1

        for e_wikiid, p_e_m in mention_ent_cand[m].items():
            if p_e_m > top_p:
                top_ent = e_wikiid
                top_p = p_e_m

        return self.is_person(top_ent)

    def build_coreference_dataset(self, dataset_lines, banner):
        """
        banner = 'name of a evaluation dataset'
        datasets = {
            'doc_id' : [line1, line2, line4, ...]
        }
        """
        if not self.args.coref:
            return dataset_lines

        # -- Create new entity candidates
        coref_dataset_lines = dict()

        for doc_id, lines_map in dataset_lines.items():
            coref_dataset_lines[doc_id] = list()

            # -- Collect entity candidates for each mention.
            mention_ent_cand = dict()
            for sample_line in lines_map:
                parts = sample_line.split("\t")
                assert doc_id == parts[0]

                mention = parts[2].lower()
                if mention not in mention_ent_cand:
                    mention_ent_cand[mention] = dict()

                assert parts[5] == 'CANDIDATES'

                if parts[6] != 'EMPTYCAND':
                    num_cand = 1
                    while parts[5 + num_cand] != 'GT:':
                        cand_parts = parts[5 + num_cand].split(',')
                        cand_ent_wikiid = int(cand_parts[0])
                        cand_p_e_m = float(cand_parts[1])

                        assert cand_p_e_m >= 0, cand_p_e_m
                        assert cand_ent_wikiid > 0
                        mention_ent_cand[mention][cand_ent_wikiid] = cand_p_e_m
                        num_cand += 1

            # -- Find coreferent mentions
            for sample_line in lines_map:
                parts = sample_line.split("\t")
                assert doc_id == parts[0]
                mention = parts[2].lower()
                assert mention in mention_ent_cand
                assert parts[-2] == 'GT:'

                # -- Grd trth infos
                grd_trth_parts = parts[-1].split(',')
                grd_trth_idx = int(grd_trth_parts[0])
                assert grd_trth_idx == -1 or len(grd_trth_parts) >= 4, sample_line
                grd_trth_entwikiid = -1

                if len(grd_trth_parts) >= 3:
                    grd_trth_entwikiid = int(grd_trth_parts[1])

                # -- Merge lists of entity candidates
                added_list = dict()
                num_added_mentions = 0

                stupid_pattern = mention.replace('%.', '%%%.').replace('%-', '%%%-')
                for m in mention_ent_cand:
                    if m != mention and ((' ' + stupid_pattern) in m or (stupid_pattern + ' ') in m) and \
                       self.mention_refers_to_person(m, mention_ent_cand):

                        if banner == 'aida_testB':
                            print('coref mention = ' + m +
                                  ' replaces original mention = ' + mention +
                                  ' ; DOC = ' + doc_id)

                        num_added_mentions += 1

                        for e_wikiid, p_e_m in mention_ent_cand[m].items():
                            if e_wikiid not in added_list:
                                added_list[e_wikiid] = 0.0
                            added_list[e_wikiid] += p_e_m

                # -- Average:
                for e_wikiid in added_list:
                    added_list[e_wikiid] /= num_added_mentions

                # -- Merge the two lists
                merged_list = mention_ent_cand[mention]
                if num_added_mentions > 0:
                    merged_list = added_list

                sorted_list = dict(sorted(merged_list.items(), key=lambda x:x[1], reverse=True))

                s = '\t'.join(parts[:6]) + '\t'

                if len(sorted_list) == 0:
                    s += 'EMPTYCAND\tGT:\t-1'
                    # **YD** "unk_ent_wikiid", "get_ent_name_from_wikiid" has been implemented
                    if grd_trth_entwikiid != self.ent_name_id.unk_ent_wikiid:
                        s += ',' + str(grd_trth_entwikiid) + ',' + \
                             self.ent_name_id.get_ent_name_from_wikiid(grd_trth_entwikiid)
                else:
                    candidates = []
                    gt_pos = -1
                    for i, (ent_wikiid, p) in enumerate(sorted_list.items()):
                        if i < 100:
                            # **YD** "get_ent_name_from_wikiid" has been implemented
                            candidates.append(str(ent_wikiid) + ',' + '{:.3f}'.format(p) + ',' +
                                              self.ent_name_id.get_ent_name_from_wikiid(grd_trth_entwikiid))
                            if ent_wikiid == grd_trth_entwikiid:
                                gt_pos = i
                        else:
                            break
                    s += '\t'.join(candidates) + '\tGT:\t'

                    if gt_pos >= 0:
                        s += str(gt_pos) + ',' + candidates[gt_pos]
                    else:
                        s += '-1'
                        # **YD** "get_ent_name_from_wikiid", "unk_ent_wikiid", has been implemented
                        if grd_trth_entwikiid != self.ent_name_id.unk_ent_wikiid:
                            s += ',' + str(grd_trth_entwikiid) + ',' + \
                                 self.ent_name_id.get_ent_name_from_wikiid(grd_trth_entwikiid)

                coref_dataset_lines[doc_id].append(s)

        assert len(dataset_lines) == len(coref_dataset_lines)
        for doc_id, _ in dataset_lines.items():
            assert len(dataset_lines[doc_id]) == len(coref_dataset_lines[doc_id]), doc_id

        return coref_dataset_lines