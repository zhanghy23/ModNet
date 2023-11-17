import os
import random
import thop
import torch
from models.crnet import crnet,blnet

from utils import logger
from utils.logger import line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(args):
    # Model loading
    if args.phase==1:
        model = crnet(phase=1)
        image1 = torch.randn([1, 2, 128, 128])
        flops, params = thop.profile(model, inputs=(image1,), verbose=False)#利用前向传输计算参数量
        flops, params = thop.clever_format([flops, params], "%.3f")
    else:
        model = blnet()
        image1 = torch.randn([1, 2, 128, 128])
        image2 = torch.randn([1, 2, 128, 128])
        flops, params = thop.profile(model, inputs=(image1,image2), verbose=False)#利用前向传输计算参数量
        flops, params = thop.clever_format([flops, params], "%.3f")
    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained)
        state_dict = torch.load(args.pretrained,
                                map_location=torch.device('cpu'))['state_dict']
        model.load_state_dict(state_dict)
        logger.info("pretrained model loaded from {}".format(args.pretrained))

    # Model flops and params counting


    # Model info logging
    logger.info(f'=> Model Name: qikunet [pretrained: {args.pretrained}]')
    #logger.info(f'=> Model Config: compression ratio=1/{args.cr}')
    # logger.info(f'=> Model Flops: {flops}')
    # logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
