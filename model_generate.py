#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress

from .effdet import create_model, create_evaluator, create_dataset, create_loader
from .effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging

try:
    from timm.layers import set_layer_config
except ImportError:
    from timm.models.layers import set_layer_config

has_apex = False
try:
    from apex import amp

    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

saved_state = None


def setup_before_inference():
    global saved_state
    saved_state = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True


def restore_after_inference():
    torch.backends.cudnn.benchmark = saved_state


def load_model(amp: bool = False, native_amp: bool = False, apex_amp: bool = False, pretrained: bool = False,
               checkpoint: str = '', torchscript: bool = False, img_size: int = None, model: str = 'tf_efficientdet_d1',
               num_classes: int = None, use_ema: bool = False, torchcompile: str = None, num_gpu: int = 1):
    setup_default_logging()

    if amp:
        if has_native_amp:
            native_amp = True
        elif has_apex:
            apex_amp = True
    assert not apex_amp or not native_amp, "Only one AMP mode should be set."
    pretrained = pretrained or not checkpoint  # might as well try to validate something

    # create model
    with set_layer_config(scriptable=torchscript):
        extra_args = {}
        if img_size is not None:
            extra_args = dict(image_size=(img_size, img_size))
        bench = create_model(
            model,
            bench_task='predict',
            num_classes=num_classes,
            pretrained=pretrained,
            redundant_bias=None,
            soft_nms=None,
            checkpoint_path=checkpoint,
            checkpoint_ema=use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (model, param_count))

    bench = bench.cuda()

    if torchscript:
        assert not apex_amp, \
            'Cannot use APEX AMP with torchscripted model, force native amp with `--native-amp` flag'
        bench = torch.jit.script(bench)
    elif torchcompile:
        bench = torch.compile(bench, backend=torchcompile)

    amp_autocast = suppress
    if apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(num_gpu)))

    bench.eval()

    return bench
