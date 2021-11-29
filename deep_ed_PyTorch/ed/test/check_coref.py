import os
import argparse

from coref_persons import CorefPersons


def test(args):
    coref_persons_cls = CorefPersons(args)

    file = os.path.join(args.root_data_dir, 'generated/test_train_data/aida_testB.csv')
    all_doc_lines = dict()

    with open(file, 'r') as reader:
        for line in reader:
            line = line.rstrip()
            parts = line.split('\t')
            doc_name = parts[0]
            if doc_name not in all_doc_lines:
                all_doc_lines[doc_name] = list()
            all_doc_lines[doc_name].append(line)

    # -- Gather coreferent mentions to increase accuracy.
    coref_persons_cls.build_coreference_dataset(all_doc_lines, 'aida_testB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='parser for ed model',
        allow_abbrev=False,
    )

    parser.add_argument(
        '--root_data_dir',
        type=str,
        default='/scratch365/yding4/EL_resource/data/deep_ed_PyTorch_data/',
        help='Root path of the data, $DATA_PATH.',
    )

    args = parser.parse_args()
    args.coref = True
    test(args)

