import time

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    start_time = time.time()
    preds = []
    gt = []
    correct = 0
    total = 0

    label_list = [i for i in range(9)]
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))

    for batch, (data, labels) in pbar:
        data, labels = data.float().to(device), labels.to(device)
        outputs = model(data)
        outputs = torch.squeeze(outputs)

        _, pred = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        preds += pred.to("cpu").tolist()
        gt += labels.to("cpu").tolist()
        pbar.update()
        pbar.set_description(
            f"Acc: {(correct / total) * 100:.2f}% "
            f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
        )
        
    consumed_time = time.time() - start_time
    accuracy = correct / total
    f1 = f1_score(
        y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
    )
    
    return f1, accuracy, consumed_time
