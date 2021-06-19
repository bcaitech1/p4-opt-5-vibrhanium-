"""PyTorch trainer module.

- Author: Jongkuk Lim, Junghoon Kim
- Contact: lim.jeikei@gmail.com, placidus36@gmail.com
"""

import wandb
import optuna
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch

from .train_utils import save_model


class TorchTrainer:
    """Pytorch Trainer."""

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        macs,
        scaler = None,
        device = "cpu",
        model_path = None,
        verbose = 1,
        cur_time = -1,
        number = -1
    ):
        """Initialize TorchTrainer class.

        Args:
            model: model to train
            criterion: loss function module
            optimizer: optimization module
            device: torch device
            verbose: verbosity level.
            best_f1: best f1 score you've ever trialed.
        """

        self.model = model
        self.model_path = model_path
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.macs = macs
        self.verbose = verbose

        if cur_time != -1:    
            self.cur_time = cur_time
            self.number = number
            self.wandb_logging = True
        else:
            self.wandb_logging = False


    def train(self, train_dataloader, val_dataloader, n_epoch, trial = None):
        """Train model.

        Args:
            train_dataloader: data loader module which is a iterator that returns (data, labels)
            n_epoch: number of total epochs for training
            val_dataloader: dataloader for validation

        Returns:
            loss and accuracy
        """
        if self.wandb_logging:
            run = wandb.init(project='OPT', name = f'{self.cur_time}_{self.number}_epochs' , reinit=False)
            
        best_acc = -1.0
        best_f1 = -1.0
        num_classes = 9
        label_list = [i for i in range(num_classes)]

        for epoch in range(n_epoch):
            running_loss, correct, total = 0.0, 0, 0
            preds, gt = [], []
            pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            self.model.train()
            for batch, (data, labels) in pbar:
                data, labels = data.float().to(self.device), labels.to(self.device)

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                else:
                    outputs = self.model(data)
                outputs = torch.squeeze(outputs)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                self.scheduler.step()

                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                preds += pred.to("cpu").tolist()
                gt += labels.to("cpu").tolist()

                running_loss += loss.item()
                pbar.update()
                pbar.set_description(
                    f"Train: [{epoch + 1:03d}] "
                    f"Loss: {(running_loss / (batch + 1)):.3f}, "
                    f"F1: {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}, "
                    f"Acc: {(correct / total) * 100:.2f}% "
                )
                if self.wandb_logging:
                    wandb.log({
                        'loss':(running_loss / (batch + 1)), 
                        'train_f1': f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0),
                        'train_acc': (correct / total)
                        })
            pbar.close()

            _, test_f1, test_acc = self.test(
                model=self.model, test_dataloader=val_dataloader
            )
            
            if self.wandb_logging:
                wandb.log({'test_f1':test_f1, 'test_acc': test_acc})

            if trial:
                trial.report(test_f1, epoch)

                if trial.should_prune():
                    if self.wandb_logging:
                        run.finish()
                    raise optuna.exceptions.TrialPruned()

            if best_f1 >= test_f1:
                continue

            best_f1 = test_f1
            best_acc = test_acc

            if self.model_path:
                print(f"Model saved. Current best test f1: {best_f1:.3f} / test acc: {best_acc:.3f}")
                save_model(
                    model=self.model,
                    path=self.model_path
                )
        if self.wandb_logging:
            run.finish()

        return best_f1, best_acc


    @torch.no_grad()
    def test(self, model, test_dataloader):
        """Test model.

        Args:
            test_dataloader: test data loader module which is a iterator that returns (data, labels)

        Returns:
            loss, f1, accuracy
        """

        running_loss = 0.0
        preds = []
        gt = []
        correct = 0
        total = 0

        num_classes = 9
        label_list = [i for i in range(num_classes)]

        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        model.to(self.device)
        model.eval()
        for batch, (data, labels) in pbar:
            data, labels = data.float().to(self.device), labels.to(self.device)

            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
            else:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            running_loss += self.criterion(outputs, labels).item()

            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            preds += pred.to("cpu").tolist()
            gt += labels.to("cpu").tolist()
            pbar.update()
            pbar.set_description(
                f" Val: {'':5} Loss: {(running_loss / (batch + 1)):.3f}, "
                f"Acc: {(correct / total) * 100:.2f}% "
                f"F1(macro): {f1_score(y_true=gt, y_pred=preds, labels=label_list, average='macro', zero_division=0):.2f}"
            )
        loss = running_loss / len(test_dataloader)
        accuracy = correct / total
        f1 = f1_score(
            y_true=gt, y_pred=preds, labels=label_list, average="macro", zero_division=0
        )
        return loss, f1, accuracy

