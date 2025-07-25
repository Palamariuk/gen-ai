import torch
from torch.utils.tensorboard import SummaryWriter

import os


class Trainer:
    def __init__(self, model, data_module,
                 log_dir='./tensorboard/logs', checkpoint_dir='./checkpoints', device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.data_module = data_module
        self.total_epochs = 0
        self.device = device
        self.model.set_device(self.device)
        self.optimizers = self.model.configure_optimizers()

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_metric = float("inf")
        self.best_epoch = -1

        self.train_loader = data_module.train_dataloader()
        self.val_loader = data_module.val_dataloader()
        self.test_loader = data_module.test_dataloader()

        self.writer = SummaryWriter(log_dir=log_dir)
        # self.writer.add_graph(model)

    def train(self, num_epochs, save_best=True, monitor_metric="loss"):
        for epoch in range(num_epochs):
            self.model.current_epoch = self.total_epochs

            print(f"Epoch {self.model.current_epoch}:")

            train_metrics = self.train_one_epoch()
            print(f"\tTrain Metrics = {train_metrics}")

            # Log epoch-level metrics to TensorBoard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"{key}/Epoch_Train", value, self.model.current_epoch)

            # Validation loop
            val_metrics = self.validate_one_epoch()
            val_loss = val_metrics.get(monitor_metric, float("inf"))
            print(f"\tVal Metrics = {val_metrics}")

            # Log epoch-level metrics to TensorBoard
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"{key}/Epoch_Validation", value, self.model.current_epoch)

            # Save the best checkpoint
            if save_best and val_loss < self.best_metric:
                self.best_metric = val_loss
                self.best_epoch = self.model.current_epoch
                self.save_checkpoint(prefix="best_")
                print(
                    f"\tBest model saved at epoch {self.model.current_epoch} with {monitor_metric}={self.best_metric:.6f}")

            self.total_epochs += 1

    def train_one_epoch(self):
        self.model.train()
        metrics_sum = {}
        num_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            step_output = self.model.training_step(batch, self.optimizers)

            for key, value in step_output.items():
                if key not in metrics_sum:
                    metrics_sum[key] = 0.0
                metrics_sum[key] += value

            if batch_idx % 10 == 0:
                for key, value in step_output.items():
                    self.writer.add_scalar(f"{key}/Train", value, self.model.current_epoch * num_batches + batch_idx)

        metrics_avg = {key: value / num_batches for key, value in metrics_sum.items()}
        return metrics_avg

    def validate_one_epoch(self):
        self.model.eval()
        metrics_sum = {}
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                step_output = self.model.validation_step(batch, batch_idx)

                for key, value in step_output.items():
                    if "img" in key:
                        self.writer.add_image(f"{key}/Validation", value, self.model.current_epoch)
                    else:
                        if key not in metrics_sum:
                            metrics_sum[key] = 0.0
                        metrics_sum[key] += value

                        self.writer.add_scalar(f"{key}/Validation", value,
                                               self.model.current_epoch * num_batches + batch_idx)

        metrics_avg = {key: value / num_batches for key, value in metrics_sum.items()}
        return metrics_avg

    def test(self):
        self.model.eval()
        metrics_sum = {}
        num_batches = len(self.test_loader)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                step_output = self.model.validation_step(batch, batch_idx)

                for key, value in step_output.items():
                    if key not in metrics_sum:
                        metrics_sum[key] = 0.0
                    metrics_sum[key] += value

        metrics_avg = {key: value / num_batches for key, value in metrics_sum.items()}
        print(f"Test Metrics: {metrics_avg}")
        return metrics_avg

    def save_checkpoint(self, prefix=''):
        best_model_path = os.path.join(self.checkpoint_dir, f"{prefix}model_epoch_{self.model.current_epoch}.pth")
        torch.save(self.model.state_dict(), best_model_path)
