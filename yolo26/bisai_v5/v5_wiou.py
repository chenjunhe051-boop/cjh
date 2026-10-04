# -*- coding: utf-8 -*-
"""WIoU v3 (Wise-IoU) 动态聚焦回归损失 —— 论文 arXiv:2301.10051。

动机：比赛的指标是 mAP50-95，而我们 mAP50(0.97) 与 mAP50-95(0.81) 差距大 =
"找得到但框不准"。WIoU v3 的非单调聚焦机制把回归梯度集中在普通质量锚框上，
屏蔽已拟合的高质量框和疑似标注噪声的低质量框，直接提升精细定位。

集成方式：包装 ultralytics.utils.loss.bbox_iou —— BboxLoss 里
loss_iou = ((1 - iou) * weight).sum()，返回 1 - L_wiou 即可无缝替换。
"""
import torch
import torch.nn as nn


class WiseIoU3(nn.Module):
    """WIoU v3: r = beta / (delta * alpha^(beta-delta)), beta = L_iou / running_mean."""

    def __init__(self, alpha=1.9, delta=3.0, momentum=0.9):
        super().__init__()
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.m = float(momentum)
        self.running_mean = 1.0   # 纯 float，设备无关（本模块不在 model 内，buffer 不会随 .cuda() 迁移）

    def forward(self, pred, target):
        # pred/target: (N, 4) xyxy
        px1, py1, px2, py2 = pred.unbind(-1)
        tx1, ty1, tx2, ty2 = target.unbind(-1)
        iw = (torch.minimum(px2, tx2) - torch.maximum(px1, tx1)).clamp(min=0)
        ih = (torch.minimum(py2, ty2) - torch.maximum(py1, ty1)).clamp(min=0)
        inter = iw * ih
        ap = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        at = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = (ap + at - inter).clamp(min=1e-7)
        iou = inter / union
        L_iou = 1.0 - iou

        Wg = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).detach()
        Hg = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).detach()
        cpx, cpy = (px1 + px2) / 2, (py1 + py2) / 2
        ctx, cty = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        R = torch.exp(((cpx - ctx) ** 2 + (cpy - cty) ** 2) / (Wg ** 2 + Hg ** 2).clamp(min=1e-7))
        L_v1 = R * L_iou

        beta = L_iou.detach() / max(self.running_mean, 1e-7)
        r = beta / (self.delta * self.alpha ** (beta - self.delta))
        loss = (r * L_v1).mean()

        with torch.no_grad():
            self.running_mean = (1 - self.m) * self.running_mean \
                + self.m * float(L_iou.mean().detach())
        return loss


def install_wiou(alpha=1.9, delta=3.0):
    """把 v8DetectionLoss 里的 CIoU 换成 WIoU v3（对其余代码零侵入）。"""
    import ultralytics.utils.loss as L
    if getattr(L.bbox_iou, "_v5_wiou", False):
        return
    wiou_loss = WiseIoU3(alpha, delta)

    def _wiou_bbox_iou(pred_bboxes, target_bboxes, *args, **kwargs):
        return 1.0 - wiou_loss(pred_bboxes, target_bboxes)   # loss_iou = 1 - iou = L_wiou

    _wiou_bbox_iou._v5_wiou = True
    L.bbox_iou = _wiou_bbox_iou
    print("[v5_wiou] CIoU -> WIoU v3 (alpha=%.1f delta=%.1f)，聚焦精细定位" % (alpha, delta))
