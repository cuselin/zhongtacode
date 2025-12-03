import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

def train_one_epoch(model, data_loader, optimizer, device, epoch, log_writer=None, num_classes=3):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    count = 0

    for i, samples in tqdm(enumerate(data_loader), desc=f"[E1] Epoch {epoch}"):
        images, targets = samples['img'], samples['label']
        images = images.to(device)
        targets = targets.to(device)

        # 三分类：若出现越界标签，明确报错（便于发现数据/配置问题）
        if targets.max().item() >= num_classes:
            raise ValueError(f"[E1] Found target id {targets.max().item()} >= num_classes={num_classes}. "
                             f"Adjust --nb_classes or remap masks.")

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

        if (i + 1) % 10 == 0:
            print(f"[E1][Epoch {epoch}] step {i+1}/{len(data_loader)}, loss={loss.item():.4f}")

    epoch_loss = running_loss / max(count, 1)
    if log_writer is not None:
        log_writer.add_scalar('train/loss', epoch_loss, epoch)
    return {'loss': epoch_loss}

@torch.no_grad()
def evaluate(model, data_loader, device, num_classes=3):
    model.eval()
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    # 正确使用 DataLoader 返回的字典 samples
    for samples in data_loader:
        images, targets = samples['img'], samples['label']
        images = images.to(device)
        targets = targets.to(device)

        # 评估阶段容错：将越界标签裁至合法范围，避免崩溃
        targets = targets.clamp_max(num_classes - 1)

        logits = model(images)
        pred = logits.argmax(dim=1)

        mask = (targets >= 0) & (targets < num_classes)
        label = num_classes * targets[mask].view(-1) + pred[mask].view(-1)
        binc = torch.bincount(label, minlength=num_classes**2)
        conf += binc.reshape(num_classes, num_classes)

    # 指标计算
    tp = conf.diag()
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    tn = conf.sum() - (tp + fp + fn)

    # OA
    oa = tp.sum().float() / conf.sum().float()

    # IoU per class
    iou = tp.float() / (tp + fp + fn).float().clamp(min=1)
    miou = iou.mean().item()

    # F1 per class
    precision = tp.float() / (tp + fp).float().clamp(min=1)
    recall = tp.float() / (tp + fn).float().clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)
    mf1 = f1.mean().item()

    stats = {
        'OA': oa.item(),
        'mIoU': miou,
        'mF1': mf1,
    }
    print(f"[E1][Eval] OA={stats['OA']:.4f}, mIoU={stats['mIoU']:.4f}, mF1={stats['mF1']:.4f}")
    return stats