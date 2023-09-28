import torch


def kernel_function(pred, target, mask, args):
    # pred:[N, L, dim]; L = 14*14 patch_num; dim = H * W * C Patch pixel number
    if args.kernel == 'RBF':
        loss = (pred - target) ** 2
        loss = loss.sum(dim=-1)  # [N, L] sum over per patch
        loss = (loss * mask).sum(-1)  # [N] exclude unmask patch
        loss = torch.exp(-(args.gamma * loss)).mean()
    elif args.kernel == 'IMQ':
        loss = (pred - target) ** 2
        loss = loss.sum(dim=-1)  # [N, L] sum over per patch
        loss = (loss * mask).sum(-1)  # [N] exclude unmask patch
        loss = args.gamma / ((args.gamma ** 2 + loss) ** 0.5)  # 0 to -1
        loss = loss.mean()
    elif args.kernel == 'laplacian':
        # \exp(-\args.gamma ||x-y||)
        loss = (pred - target) ** 2
        loss = loss.sum(dim=-1)  # [N, L] sum over per patch
        loss = (loss * mask).sum(-1)  # [N] exclude unmask patch
        loss = torch.exp(- args.gamma * (loss ** 0.5)).mean()
    else:
        AssertionError('wrong kernel:' + args.kernel)

    return -loss
