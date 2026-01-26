import os
from typing import Optional, Any
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils.progress_bar import display_progress_bar
from utils.early_stoping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _run_epoch(self, loader, training=True, show_progress=True):
        self.model.train() if training else self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_targets = [], []
        num_batches = len(loader)

        with torch.set_grad_enabled(training):
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if training:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                running_loss += loss.item()
                _, preds = outputs.max(1)
                total += targets.size(0)
                correct += preds.eq(targets).sum().item()

                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())

                if show_progress:
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = 100.0 * correct / total

                    display_progress_bar(
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        loss=current_loss,
                        accuracy=current_acc,
                    )

        epoch_loss = running_loss / num_batches
        epoch_acc = 100.0 * accuracy_score(all_targets, all_preds)

        return epoch_loss, epoch_acc, all_preds, all_targets

    def fit(
        self,
        trainloader,
        valloader,
        epochs=10,
        patience=20,
        delta=1e-4,
        model_name="model",
        topk=5,
    ):
        best_vloss = float("inf")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_time = time.time()
        early_stopping = EarlyStopping(patience=patience, delta=delta)
        topk_models = []

        for epoch in range(epochs):
            model_path = f"{model_name}_e{epoch}_{timestamp}.pt"

            print(f"\nEpoch {epoch + 1}/{epochs} - Training")
            t_loss, t_acc, _, _ = self._run_epoch(
                trainloader, training=True, show_progress=True
            )
            v_loss, v_acc, _, _ = self._run_epoch(
                valloader, training=False, show_progress=True
            )

            if self.scheduler:
                self.scheduler.step()

            # Log Metrics
            self.history["train_loss"].append(t_loss)
            self.history["train_acc"].append(t_acc)
            self.history["val_loss"].append(v_loss)
            self.history["val_acc"].append(v_acc)

            # Top K model saved
            if v_loss < best_vloss:
                best_vloss = v_loss
                best_model_weights = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                torch.save(best_model_weights, model_path)
                print(
                    f"\n--> NEW RECORD: Best model saved (Val loss: {best_vloss:.4f})"
                )
                topk_models.append((v_loss, model_path))
                topk_models.sort(key=lambda x: x[0])

                if topk_models is not None and len(topk_models) > topk:
                    _, worst_path = topk_models.pop(-1)
                    if os.path.exist(worst_path):
                        os.remove(worst_path)

            # Early Stopping
            early_stopping(val_loss=v_loss, model=self.model)

            if early_stopping.early_stop:
                print(
                    f"\nEarly stopping at epoch {epoch+1} with best validation loss: {early_stopping.best_loss:.4f}"
                )
                early_stopping.load_best_model(self.model)
                break

            print(f"\nTrain Loss: {t_loss:.4f}, Train Acc: {t_acc:.2f}%")
            print(f"Validation Loss: {v_loss:.4f}, Validation Acc: {v_acc:.2f}%")

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")

    def evaluate(self, testloader, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        loss, acc, preds, targets = self._run_epoch(
            testloader, training=False, show_progress=True
        )

        metrics = {
            "loss": loss,
            "accuracy": acc / 100,
            "precision": precision_score(
                targets, preds, average="weighted", zero_division=0
            ),
            "f1": f1_score(targets, preds, average="weighted", zero_division=0),
        }

        print("\n" + "=" * 25 + "\nTEST SET RESULTS\n" + "=" * 25)

        for k, v in metrics.items():
            print(f"{k.capitalize():<10}: {v:.4f}")

        return preds, targets, metrics
