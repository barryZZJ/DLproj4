import torch

def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除

    if torch.is_tensor(output):
        output = output.data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()

    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)

def dice_coef_gpu(output, target):
    eps = 1e-5
    N = target.size(0)
    output_flat = output.view(N, -1)
    target_flat = target.view(N, -1)
    
    tp = torch.sum(output_flat * target_flat, dim=1)
    fp = torch.sum(output_flat, dim=1) - tp
    fn = torch.sum(target_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)

    return loss.sum() / N


