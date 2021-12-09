import argparse


def produce(args):
    return args.t * 2


class test(object):
    def __init__(self, args):
        self.args = args
        args.t = 100
        args.g = produce(args)
        print(args)
        print(self.args)


def main():
    parser = argparse.ArgumentParser(
        description='test entity embedding model_a',
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
        '--optimization',
        type=str,
        default='ADAGRAD',
        choices=['RMSPROP', 'ADAGRAD', 'ADAM', 'SGD'],
        help='optimizer type',
    )

    args = parser.parse_args()

    tt = test(args)
    print("after passing", args)


if __name__ == '__main__':
    main()
