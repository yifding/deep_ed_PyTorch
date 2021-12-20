import os
import argparse

from deep_ed_PyTorch.entities.relatedness import REWTR


class FilterWikiHyperlinkContextsRLTD(object):
    def __init__(self, args):
        self.args = args

        self.input = 'generated/wiki_hyperlink_contexts.csv'
        self.input_file = os.path.join(args.root_data_dir, self.input)

        self.output = 'generated/wiki_hyperlink_contexts_RLTD.csv'
        self.output_file = os.path.join(args.root_data_dir, self.output)

        # **YD** self.rewtr not implemented
        self.rewtr = REWTR(args)

        self.read_and_write()

    def read_and_write(self):
        print('\nStarting dataset filtering.')
        num_lines = 0

        ouf = open(self.output_file, 'w')

        with open(self.input_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\n\t')
                num_lines += 1
                if num_lines % 50000 == 0:
                    print('    =======> processed ' + str(num_lines) + ' lines')

                parts = line.split('\t')

                assert (parts[len(parts) - 2] == 'GT:')
                grd_str = parts[len(parts) - 1]

                grd_str_parts = grd_str.split(',')

                grd_pos = int(grd_str_parts[0])
                grd_ent_wikiid = int(grd_str_parts[1])

                assert grd_pos >= 0, 'invalid grd_pos'
                assert grd_ent_wikiid > 0, 'invalid entity wikiid'

                if grd_ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                    assert (parts[5] == 'CANDIDATES')

                    output_line = '\t'.join(parts[:6]) + '\t'
                    new_grd_pos = -1
                    new_grd_str_without_idx = None

                    i = 1
                    added_ents = 0

                    while parts[5 + i] != 'GT:':
                        s = parts[5 + i]
                        s_parts = s.split(',')
                        ent_wikiid = int(s_parts[0])
                        if ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                            added_ents += 1
                            output_line = output_line + s + '\t'

                        if i - 1 == grd_pos:
                            assert ent_wikiid == grd_ent_wikiid, 'Error for: ' + line
                            new_grd_pos = added_ents - 1
                            new_grd_str_without_idx = s
                        i += 1

                    assert new_grd_pos >= 0
                    output_line += 'GT:\t' + str(new_grd_pos) + ',' + new_grd_str_without_idx
                    ouf.write(output_line + '\n')

        ouf.close()

        print('----> Dataset filtering done.')


def test(args):
    FilterWikiHyperlinkContextsRLTD(args)


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
    parser.add_argument(
        '--rltd_entity_source',
        type=str,
        choices=["ed_train", "entity_name_files"],
        default="ed_train",
        help='rltd_entity_source to inject entity set.',
    )

    parser.add_argument(
        '--rltd_ed_train_files',
        type=eval,
        default="['aida_train.csv','aida_testA.csv','aida_testB.csv',"
                "'wned-aquaint.csv','wned-msnbc.csv','wned-ace2004.csv','wned-clueweb.csv','wned-wikipedia.csv']",
        help='rltd_ed_train_files',
    )

    parser.add_argument(
        '--rltd_entity_name_files',
        type=eval,
        default="[]",
        help='rltd_entity_name_files',
    )

    args = parser.parse_args()
    print(args)
    test(args)