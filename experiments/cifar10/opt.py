import torch
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import LitResNet, CIFAR10DataModule
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
from optforget import FineTuneForgetProblem, StochasticMSASolver, MSAOptimizer

sns.set_style("whitegrid")
pl.seed_everything(42, workers=True)
PRETRAINED_CKPT_PATH = 'checkpoints/pretrained-best.ckpt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Setting up data modules...")
data_module = CIFAR10DataModule(batch_size=256)
data_module.setup()
val_loaders_A_and_B = data_module.val_dataloader()
print("✅ Data modules are ready.")


class ForgettingTracker(pl.Callback):

    def __init__(self):
        self.history = []
        self.epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        acc_A = trainer.callback_metrics.get('val_acc_A')
        acc_B = trainer.callback_metrics.get('val_acc_B')

        if acc_A is not None and acc_B is not None:
            self.history.append({
                'epoch': self.epoch,
                'Task A Acc': acc_A.item() * 100,
                'Task B Acc': acc_B.item() * 100,
            })
        self.epoch = self.epoch + 1


forgetting_tracker = ForgettingTracker()

print("\n--- Phase 1: Pre-training on Task A (classes 0-4) ---")
model = LitResNet(datamodule=data_module,
                  learning_rate=1e-3,
                  optimizer_name='AdamW')
trainer_pretrain = pl.Trainer(max_epochs=1,
                              accelerator='auto',
                              callbacks=[forgetting_tracker],
                              deterministic=True,
                              enable_checkpointing=False,
                              logger=False,
                              enable_progress_bar=True)
trainer_pretrain.fit(model,
                     train_dataloaders=data_module.train_dataloader(task='A'),
                     val_dataloaders=val_loaders_A_and_B)
print("\n✅ Pre-training on Task A finished.")

print("\n\n--- Phase 2: Fine-tuning on Task B with SGD ---")
model.hparams.optimizer_name = 'MSAOptimizer'
model.hparams.lambda_reg = 0.01
model.hparams.c1 = 1.0
model.hparams.c2 = 1.0
model.hparams.eta = 0.01
model.hparams.msa_num_steps = 5
model.hparams.t0 = 0.0
model.hparams.tf = 1.0
model.flat_params = torch.nn.Parameter(
    parameters_to_vector(model.model.parameters()).detach())
forgetting_tracker.epoch = forgetting_tracker.epoch - 1
trainer_finetune = pl.Trainer(max_epochs=5,
                              accelerator='auto',
                              callbacks=[forgetting_tracker],
                              deterministic=True,
                              logger=False,
                              enable_progress_bar=True)

trainer_finetune.fit(model,
                     train_dataloaders=data_module.train_dataloader(task='B'),
                     val_dataloaders=val_loaders_A_and_B)
print("\n✅ Fine-tuning finished.")
