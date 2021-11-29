import os
import argparse

# **YD** be very careful about the index of e_freq, may change th logic to python like.
# example:
# 3434750 United States   1347056
# then:
# self.e_freq['ent_f_start'][3434750] = 1
# self.e_freq['ent_f_end'][3434750] = 1347056
# self.get_ent_freq('3434750') = self.e_freq['ent_f_end'][3434750]  - self.e_freq['ent_f_start']['3434750'] + 1


class EFreqIndex(object):
    def __init__(self, args, add_rewtr=False):
        self.args = args

        self.ent_freq = 'generated/ent_wiki_freq.txt'
        self.ent_freq_file = os.path.join(self.args.root_data_dir, self.ent_freq)

        # **YD** rewtr logic will be finished in the future
        if add_rewtr:
            self.rewtr = None

        self.e_freq = dict()
        self.load_from_file()

    @property
    def dict(self):
        return self.e_freq

    def load_from_file(self):
        print('==> Loading entity freq map')
        min_freq = 1

        self.e_freq['ent_f_start'] = dict()
        self.e_freq['ent_f_end'] = dict()
        self.e_freq['total_freq'] = 0
        self.e_freq['sorted'] = dict()

        cur_start = 1
        cnt = 0

        with open(self.ent_freq_file, 'r') as reader:
            for line in reader:
                line = line.rstrip('\t\n')
                parts = line.split('\t')
                ent_wikiid = int(parts[0])
                ent_f = int(parts[2])

                assert ent_wikiid > 0

                if not hasattr(self, 'rewtr') or ent_wikiid in self.rewtr.reltd_ents_wikiid_to_rltdid:
                    self.e_freq['ent_f_start'][ent_wikiid] = cur_start
                    self.e_freq['ent_f_end'][ent_wikiid] = cur_start + ent_f - 1
                    self.e_freq['sorted'][cnt] = ent_wikiid
                    cur_start = cur_start + ent_f
                    cnt += 1

        self.e_freq['total_freq'] = cur_start - 1
        print('    Done loading entity freq index. Size = ' + str(cnt))

    def get_ent_freq(self, ent_wikiid):
        assert type(ent_wikiid) is int
        if ent_wikiid in self.e_freq['ent_f_start']:
            return self.e_freq['ent_f_end'][ent_wikiid] - self.e_freq['ent_f_start'][ent_wikiid] + 1
        else:
            return 0


def test(args):
    EFreqIndex(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='evaluate the coverage percent of entity dictionary on evaluate GT entity'
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )
    args = parser.parse_args()
    test(args)


