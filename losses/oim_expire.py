import torch
import torch.nn.functional as F
from torch import autograd, nn

# from utils.distributed import tensor_gather


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, ious, reliability, temperature, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, ious, reliability, temperature, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1) * torch.exp(temperature * reliability)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, ious, reliability, temperature, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        reliability += 1
        for x, y, iou in zip(inputs, targets, ious):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
                reliability[y] = 0
            else:
                cq[header] = x
                reliability[len(lut)+header] = 0
                header = (header + 1) % cq.size(0)
                
        return grad_inputs, None, None, None, None, None, None, None, None


def oim(inputs, targets, ious, reliability, temperature, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, ious, reliability, torch.tensor(temperature), lut, cq, torch.tensor(header), torch.tensor(momentum))

        
class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, max_epoch=None):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
        self.register_buffer("reliability", torch.zeros(self.num_pids + self.num_unlabeled))
        self.header_cq = 0
        self.temperature = -1 * 0.01
        

    def forward(self, inputs, roi_label, roi_ious, epoch=None):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        ious = torch.cat(roi_ious)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        ious = ious[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, ious, self.reliability, self.temperature, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        
        projected *= self.oim_scalar
        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)

        return loss_oim