import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_ed_PyTorch.ed.args import arg_parse
from deep_ed_PyTorch.ed.test.test import Test
from deep_ed_PyTorch.ed.model.model_local import ModelLocal
from deep_ed_PyTorch.ed.model.model_global import ModelGlobal


def test(args):
    if args.model_type == 'local':
        model = ModelLocal(args)
    else:
        assert args.model_type == 'global'
        model = ModelGlobal(args)

    model_ckpt = os.path.join(args.root_data_dir, 'generated/ed_models/' + args.test_one_model_file)
    args.model = model
    model.load_state_dict(torch.load(model_ckpt))
    model.eval()

    test_cls = Test(args)
    test_cls.test()


if __name__ == '__main__':
    args = arg_parse()
    test(args)
