"""
CARTE Pretrainer.
"""

import torch
from tqdm import tqdm
from src.carte_model_dev import CARTE_Pretrain_Dev_Model


class CARTE_Pretrainer:
    def __init__(
        self,
        train_loader,
        model_configs: dict,
        save_dir: str,
        learning_rate: float = 1e-6,
        warmup_steps: int = 10000,
        device: str = "cuda:0",
        save_every: int = 1000,
    ):

        self.train_loader = train_loader
        self.model_configs = model_configs
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.device = device
        self.save_every = save_every

    def fit(self):

        self._set_pretrainer()
        self.model.train()
        step = 0

        # Iterate with number of steps defined by the train_loader
        for x, edge_attr, mask, y in tqdm(self.train_loader):
            step += 1

            # Send to device
            x = x.to(self.device_)
            edge_attr = edge_attr.to(self.device_)
            mask = mask.to(self.device_)
            y = y.to(self.device_).to(torch.float32)

            # Run step
            self._run_step(x, edge_attr, mask, y, step)

        return None

    def _run_step(self, x, edge_attr, mask, y, step):

        self.optimizer.zero_grad()  # Clear gradients.
        out = self.model(x, edge_attr, mask)  # Perform a single forward pass.
        loss = self.criterion_(out, y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        self.optimizer.step()  # Update parameters based on gradients.

        if step % self.save_every == 0:
            self.loss_.append(f"{round(loss.detach().item(), 4)}")
            self._save_checkpoint(step)

        return None

    def _save_checkpoint(self, step):

        ckp = self.model.state_dict()
        save_path = self.save_dir + f"/ckpt_step{step}.pt"
        torch.save(ckp, save_path)

        log_path = self.save_dir + f"/log_train.txt"
        with open(log_path, "w") as output:
            for row in self.loss_:
                output.write(str(row) + "\n")

    def _set_pretrainer(self):

        self.device_ = torch.device(self.device)
        self.loss_ = []

        # Model
        model = CARTE_Pretrain_Dev_Model(**self.model_configs)
        self.model = model.to(self.device_)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )
        # Scheduler - warmup / decay
        self.scheduler_warm = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1,
            total_iters=self.warmup_steps,
        )
        self.scheduler_decay = torch.optim.lr_scheduler.PolynomialLR(
            self.optimizer,
            total_iters=int(len(self.train_loader) - self.warmup_steps),
        )
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[self.scheduler_warm, self.scheduler_decay],
            milestones=[self.warmup_steps],
        )
        self.criterion_ = torch.nn.BCEWithLogitsLoss()

        return
