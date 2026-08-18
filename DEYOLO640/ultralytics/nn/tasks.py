# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import copy
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d, Focus,
                                    GhostBottleneck, GhostConv, HGBlock, HGStem, RepC3, RepConv, DEA, DEA3, C2f_BiFocus)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.loss import v8DetectionLoss
from ultralytics.yolo.utils.plotting import feature_visualization
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x1, x2=None, x3=None, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
            x1 (torch.Tensor | dict): RGB tensor, or batch dict (training).
            x2 (torch.Tensor, optional): IR tensor (inference), or None (training).
            x3 (torch.Tensor, optional): Depth tensor (inference), or None (training).

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x1, dict):
            # Training/validation: single batch dict containing img/img2/img3
            return self.loss(x1, *args, **kwargs)
        return self.predict(x1, x2, x3, *args, **kwargs)

    def predict(self, x1, x2, x3=None, profile=False, visualize=False, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x1 (torch.Tensor): The input RGB tensor.
            x2 (torch.Tensor): The input IR tensor.
            x3 (torch.Tensor, optional): The input Depth tensor.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        return self._predict_once(x1, x2, x3, profile, visualize)

    def _predict_once(self, x1, x2, x3=None, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x1 (torch.Tensor): RGB input tensor.
            x2 (torch.Tensor): IR input tensor.
            x3 (torch.Tensor, optional): Depth input tensor.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        lb = len(self.yaml['backbone'])
        lb2 = lb + len(self.yaml['backbone2'])
        has_backbone3 = 'backbone3' in self.yaml
        lb3 = lb2 + len(self.yaml['backbone3']) if has_backbone3 else lb2

        if x3 is None:  # validation/prediction without depth
            x3 = torch.zeros_like(x2)

        y, dt = [], []  # outputs
        running = x2  # running variable for head
        for m in self.model:
            if m.i < lb:
                x1 = m(x1)
                y.append(x1 if m.i in self.save else None)  # save backbone1 output
            elif m.i < lb2:
                x2 = m(x2)
                y.append(x2 if m.i in self.save else None)  # save backbone2 output
                running = x2
            elif m.i < lb3:
                x3 = m(x3)
                y.append(x3 if m.i in self.save else None)  # save backbone3 output
                running = x3
            else:
                # Head: f=-1 means "keep running", otherwise resolve from saved outputs
                if m.f != -1:
                    running = y[m.f] if isinstance(m.f, int) else [running if j == -1 else y[j] for j in m.f]
                if profile:
                    self._profile_one_layer(m, running, dt)
                running = m(running)
                y.append(running if m.i in self.save else None)

            if visualize:
                feature_visualization(running, m.type, m.i, save_dir=visualize)

        return running

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=[x.clone() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(len(self.yaml['backbone'])):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """
        Compute loss from a single batch dict.

        Args:
            batch (dict): Batch dict containing 'img', 'img2', 'img3', 'bboxes', 'cls', etc.
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        x3 = batch.get('img3')
        preds = self.forward(batch['img'], batch['img2'], x3) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, ch2=3, ch3=3, nc=None,
                 verbose=True):
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # backbone1: RGB 3ch
        ch2 = self.yaml.get('ch2', ch2)  # backbone2: IR 3ch
        ch3 = self.yaml.get('ch3', ch3)  # backbone3: Depth 3ch
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, ch2=ch2, ch3=ch3, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            has_backbone3 = 'backbone3' in self.yaml
            if has_backbone3:
                forward = lambda x, x2, x3: self.forward(x, x2, x3)
                m.stride = torch.tensor(
                    [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s),
                                                       torch.zeros(1, ch2, s, s),
                                                       torch.zeros(1, ch3, s, s))])
            else:
                forward = lambda x, x2: self.forward(x, x2)
                m.stride = torch.tensor(
                    [s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s),
                                                       torch.zeros(1, ch2, s, s))])
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """Clip YOLOv5 augmented inference tails."""
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        return v8DetectionLoss(self)


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.yolo.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cuda',weights_only=False), file  # load
    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cuda',weights_only=False), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module compatibility updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, ch2, ch3=3, verbose=True):  # model_dict, input_channels(3)
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    ch2 = [ch2]
    ch3 = [ch3]
    layers, save, c2, c22, c32 = [], [], ch[-1], ch2[-1], ch3[-1]  # layers, savelist, ch out

    has_backbone3 = 'backbone3' in d
    layers_single = len(d['backbone'])
    layers_double = layers_single + len(d['backbone2'])

    all_layers = d['backbone'] + d['backbone2']
    if has_backbone3:
        all_layers = all_layers + d['backbone3']
    all_layers = all_layers + d['head']

    for i, (f, n, m, args) in enumerate(all_layers):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain

        if i < layers_single:
            # ===== backbone1 (RGB) =====
            c_out, n_new = _parse_layer(m, n, f, args, ch, nc, max_channels, width)
            c2 = c_out
            if n_new is not None:
                n = n_new

        elif i < layers_double:
            # ===== backbone2 (IR) =====
            c_out, n_new = _parse_layer(m, n, f, args, ch2, nc, max_channels, width)
            c22 = c_out
            if n_new is not None:
                n = n_new

        elif has_backbone3 and i < layers_double + len(d['backbone3']):
            # ===== backbone3 (Depth) =====
            c_out, n_new = _parse_layer(m, n, f, args, ch3, nc, max_channels, width)
            c32 = c_out
            c22 = c32  # 让 ch2.append(c22) 用 backbone3 的值，而非 backbone2 残留值
            if n_new is not None:
                n = n_new

        else:
            # ===== head =====
            if m in (Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                     BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
                c12, c22 = ch2[f], args[0]
                if c22 != nc:
                    c22 = make_divisible(min(c22, max_channels) * width, 8)
                args = [c12, c22, *args[1:]]
                if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                    args.insert(2, n)
                    n = 1
            elif m is AIFI:
                args = [ch2[f], *args]
            elif m in (HGStem, HGBlock):
                c12, cm2, c22 = ch2[f], args[0], args[1]
                args = [c12, cm2, c22, *args[2:]]
                if m is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch2[f]]
            elif m is Concat:
                c22 = sum(ch2[x] for x in f)
            elif m is Detect:
                args.append([ch2[x] for x in f])
            elif m is C2f_BiFocus:
                c12, c22 = ch2[f], args[0]
                if c22 != nc:
                    c22 = make_divisible(min(c22, max_channels) * width, 8)
                args = [c12, c22, *args[1:]]
            elif m in (DEA, DEA3):
                c12, c22 = ch2[f[0]], args[0]
                if c22 != nc:
                    c22 = make_divisible(min(c22, max_channels) * width, 8)
                args = [c12, *args[1:]]
            else:
                c22 = ch2[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        if i < layers_single:
            ch.append(c2)
        if i == layers_single:
            ch2 = list(ch)
        if i >= layers_single:
            ch2.append(c22)
        if has_backbone3 and layers_double <= i < layers_double + len(d['backbone3']):
            if i == layers_double:
                ch3 = []
            ch3.append(c32)

    return nn.Sequential(*layers), sorted(save)


def _parse_layer(m, n, f, args, ch_list, nc, max_channels, width):
    """Parse a single backbone layer. Returns (c_out, n_new) where n_new is 1 if repeats were folded into args, else None."""
    c2 = ch_list[f]  # default output channel
    n_new = None

    if m in (Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
             BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
        c1, c2 = ch_list[f], args[0]
        if c2 != nc:
            c2 = make_divisible(min(c2, max_channels) * width, 8)
        args[:] = [c1, c2, *args[1:]]
        if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
            args.insert(2, n)  # number of repeats
            n_new = 1
    elif m is AIFI:
        args[:] = [ch_list[f], *args]
    elif m in (HGStem, HGBlock):
        c1, cm, c2 = ch_list[f], args[0], args[1]
        args[:] = [c1, cm, c2, *args[2:]]
        if m is HGBlock:
            args.insert(4, n)
            n_new = 1
    elif m is nn.BatchNorm2d:
        args[:] = [ch_list[f]]
    elif m is Concat:
        c2 = sum(ch_list[x] for x in f)
    elif m is Detect:
        args.append([ch_list[x] for x in f])
    elif m is C2f_BiFocus:
        c1, c2 = ch_list[f], args[0]
        if c2 != nc:
            c2 = make_divisible(min(c2, max_channels) * width, 8)
        args[:] = [c1, c2, *args[1:]]

    return c2, n_new


def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    guessed_scale = guess_model_scale(path)
    d['scale'] = d.get('scale') or guessed_scale or None  # preserve YAML scale if present
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''


def guess_model_task(model):
    """
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m == 'detect':
            return 'detect'

    # Guess from model cfg
    if isinstance(model, dict):
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # PyTorch model
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, Detect):
                return 'detect'

    # Guess from model filename
    if isinstance(model, (str, Path)):
        model = Path(model)
        if 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect'.")
    return 'detect'  # assume detect
