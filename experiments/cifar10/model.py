import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import timm
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchmetrics import Accuracy
import numpy as np


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './data', batch_size: int = 256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        self.task = 'A'  # 'A' for classes 0-4, 'B' for classes 5-9

    def prepare_data(self):
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        full_trainset = torchvision.datasets.CIFAR10(self.data_dir,
                                                     train=True,
                                                     transform=self.transform)
        full_testset = torchvision.datasets.CIFAR10(self.data_dir,
                                                    train=False,
                                                    transform=self.transform)

        train_targets = np.array(full_trainset.targets)
        test_targets = np.array(full_testset.targets)

        # Task A: classes 0-4
        indices_train_A = np.where(train_targets < 5)[0]
        indices_test_A = np.where(test_targets < 5)[0]
        self.train_A = Subset(full_trainset, indices_train_A)
        self.val_A = Subset(full_testset, indices_test_A)

        # Task B: classes 5-9
        indices_train_B = np.where(train_targets >= 5)[0]
        indices_test_B = np.where(test_targets >= 5)[0]
        self.train_B = Subset(full_trainset, indices_train_B)
        self.val_B = Subset(full_testset, indices_test_B)

    def train_dataloader(self, task: str = "A"):
        dataset = self.train_A if task == 'A' else self.train_B
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        """
        Returns the validation dataloaders.
        """
        return [
            DataLoader(self.val_A,
                       batch_size=self.batch_size,
                       num_workers=4,
                       pin_memory=True),
            DataLoader(self.val_B,
                       batch_size=self.batch_size,
                       num_workers=4,
                       pin_memory=True)
        ]


class LitResNet(pl.LightningModule):

    def __init__(self,
                 num_classes=10,
                 learning_rate=1e-3,
                 optimizer_name='SGD'):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model('resnet18',
                                       pretrained=False,
                                       num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3,
                                     64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)
        self.model.maxpool = nn.Identity()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_A = Accuracy(task='multiclass', num_classes=num_classes)
        self.accuracy_B = Accuracy(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        if dataloader_idx == 0:
            self.accuracy_A.update(outputs, labels)
            self.log('val_acc_A',
                     self.accuracy_A,
                     on_step=False,
                     on_epoch=True,
                     add_dataloader_idx=False)
            self.log('val_loss_A',
                     loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     add_dataloader_idx=False)
        else:
            self.accuracy_B.update(outputs, labels)
            self.log('val_acc_B',
                     self.accuracy_B,
                     on_step=False,
                     on_epoch=True,
                     add_dataloader_idx=False)
            self.log('val_loss_B',
                     loss,
                     on_step=False,
                     on_epoch=True,
                     prog_bar=True,
                     add_dataloader_idx=False)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == 'AdamW':
            return optim.AdamW(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=0.05)
        return optim.SGD(self.parameters(),
                         lr=self.hparams.learning_rate,
                         momentum=0.9)
