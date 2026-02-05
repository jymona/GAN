import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import matplotlib
# 使用非交互后端
matplotlib.use('Agg')  # 必须在导入pyplot之前
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF
from datetime import datetime

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class DatasetManager:
    """数据集管理器"""
    
    @staticmethod
    def get_dataset_info(dataset_name):
        """获取数据集信息"""
        dataset_info = {
            'mnist': {
                'num_classes': 10,
                'channels': 1,
                'img_size': (28, 28),
                'mean': (0.1307,),
                'std': (0.3081,),
                'train_transform': transforms.Compose([
                    transforms.ToTensor(),
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                ])
            },
            'cifar10': {
                'num_classes': 10,
                'channels': 3,
                'img_size': (32, 32),
                'mean': (0.4914, 0.4822, 0.4465),
                'std': (0.2470, 0.2435, 0.2616),
                'train_transform': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                ])
            },
            'cifar100': {
                'num_classes': 100,
                'channels': 3,
                'img_size': (32, 32),
                'mean': (0.5071, 0.4865, 0.4409),
                'std': (0.2673, 0.2564, 0.2762),
                'train_transform': transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                ]),
                'test_transform': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
                ])
            }
        }
        
        return dataset_info.get(dataset_name.lower())
    
    @staticmethod
    def get_celeba_transform(img_size=(64, 64)):
        """Get CelebA data transform"""
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    @classmethod
    def load_dataset(cls, dataset_name, data_root='./data', batch_size=128, 
                    train_split=0.8, img_size=None):
        """Load dataset"""
        
        if dataset_name.lower() == 'celeba':
            return cls._load_celeba_dataset(data_root, batch_size, img_size)
        
        # Get dataset info
        info = cls.get_dataset_info(dataset_name)
        if not info:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Update image size
        if img_size:
            info = info.copy()
            info['img_size'] = img_size
        
        print(f"Loading dataset: {dataset_name}")
        print(f"  Image size: {info['img_size']}")
        print(f"  Channels: {info['channels']}")
        print(f"  Number of classes: {info['num_classes']}")
        
        # Load dataset
        if dataset_name.lower() == 'mnist':
            train_dataset = datasets.MNIST(
                root=data_root, train=True, download=True,
                transform=info['train_transform']
            )
            test_dataset = datasets.MNIST(
                root=data_root, train=False, download=True,
                transform=info['test_transform']
            )
            
        elif dataset_name.lower() == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=info['train_transform']
            )
            test_dataset = datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=info['test_transform']
            )
            
        elif dataset_name.lower() == 'cifar100':
            train_dataset = datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=info['train_transform']
            )
            test_dataset = datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=info['test_transform']
            )
        
        # split train and val
        train_size = int(train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        print(f"  Train set: {len(train_dataset)} samples")
        print(f"  Validation set: {len(val_dataset)} samples")
        print(f"  Test set: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader, info
    
    @classmethod
    def _load_celeba_dataset(cls, data_root, batch_size=128, img_size=None):
        """Load CelebA dataset"""
        if img_size is None:
            img_size = (64, 64)
        
        transform = cls.get_celeba_transform(img_size)
        
        print(f"Loading dataset: CelebA")
        print(f"  Image size: {img_size}")
        print(f"  Channels: 3")
        
        try:
            # Try loading CelebA
            train_dataset = datasets.CelebA(
                root=data_root, split='train', download=True,
                transform=transform
            )
            val_dataset = datasets.CelebA(
                root=data_root, split='valid', download=True,
                transform=transform
            )
            test_dataset = datasets.CelebA(
                root=data_root, split='test', download=True,
                transform=transform
            )
            
        except Exception as e:
            print(f"Failed to load CelebA: {e}")
            print("Please ensure torchvision>=0.10.0 is installed and the dataset is accessible")
            raise
        
        # Create data loaders (CelebA dataset is small, no need to split)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        print(f"  Train set: {len(train_dataset)} samples")
        print(f"  Validation set: {len(val_dataset)} samples")
        print(f"  Test set: {len(test_dataset)} samples")
        
        # Create info dictionary
        info = {
            'num_classes': None,  # CelebA is multi-label classification
            'channels': 3,
            'img_size': img_size,
            'mean': (0.5, 0.5, 0.5),
            'std': (0.5, 0.5, 0.5)
        }
        
        return train_loader, val_loader, test_loader, info

# ==================== 配置和路径设置 ====================
class Config:
    def __init__(self, base_dir=None, exp_name=None, **kwargs):
        # 超参数
        self.batch_size = kwargs.get('batch_size', 128)
        self.epochs = kwargs.get('epochs', 50)
        self.latent_dim = kwargs.get('latent_dim', 128)
        self.hidden_dims = kwargs.get('hidden_dims', [512, 256])
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        
        # 数据集配置
        self.dataset = kwargs.get('dataset', 'cifar10')
        self.use_bce = kwargs.get('use_bce', False)  # 彩色图像通常用MSE
        self.img_size = kwargs.get('img_size', None)  # 可以自定义尺寸
        self.exp_name = exp_name
        self.base_dir = Path(base_dir) if base_dir else None
        
        # 创建存储路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"vae_experiment_{timestamp}"
        if self.base_dir is None:
            self.base_dir = Path("./experiments") / self.exp_name
        
        # 子目录
        self.model_dir = self.base_dir / "models"
        self.plot_dir = self.base_dir / "plots"
        self.sample_dir = self.base_dir / "samples"
        self.log_dir = self.base_dir / "logs"
        
        # 创建目录
        for dir_path in [self.model_dir, self.plot_dir, self.sample_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"实验目录: {self.base_dir}")

# ==================== VAE模型 ====================
class FlexibleVAE(nn.Module):
    def __init__(self, img_size=(32, 32), channels=3, hidden_dims=[512, 256], latent_dim=128, use_bce=True, dropout_rate=0.2):
        super(FlexibleVAE, self).__init__()
        
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.use_bce = use_bce
        
        self.input_dim = channels * img_size[0] * img_size[1]
        
        # encoder
        encoder_layers = []
        prev_dim = self.input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # 解码器
        decoder_layers = []
        prev_dim = latent_dim
        reversed_dims = list(reversed(hidden_dims))
        
        for hidden_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(prev_dim, self.input_dim))
        
        # 根据损失类型选择激活函数
        if use_bce:
            decoder_layers.append(nn.Sigmoid())  # BCE需要[0,1]
        else:
            decoder_layers.append(nn.Tanh())  # MSE可以使用[-1,1]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"Create VAE model:")
        print(f"  Input Dim: {self.input_dim}")
        print(f"  Image Size: {img_size}")
        print(f"  Channels: {channels}")
        print(f"  Latent Dim: {latent_dim}")
        print(f"  Loss Type: {'BCE' if use_bce else 'MSE'}")
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        recon = self.decoder(z)
        return recon.view(-1, self.channels, self.img_size[0], self.img_size[1])
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, num_samples, device):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)

# ==================== Early Stopping ====================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

# ==================== Flexible VAE Trainer ====================
class FlexibleVAETrainer:
    """
    Flexible VAE Trainer TO SUPPORT DIFFERENT DATASETS
    """
    def __init__(self, model, device, config, dataset_info):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.dataset_info = dataset_info
        
        # Set according to dataset information
        self.img_size = dataset_info['img_size']
        self.channels = dataset_info['channels']
        self.mean = torch.tensor(dataset_info['mean']).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(dataset_info['std']).view(1, -1, 1, 1).to(device)
        
        # 选择损失函数
        if config.use_bce:
            self.loss_fn = self.vae_loss_bce
            print("Using BCE loss function")
        else:
            self.loss_fn = self.vae_loss_mse
            print("Using MSE loss function")
        
        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=15, min_delta=0.001)
        
        # 记录历史
        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'learning_rates': []
        }
    
    def vae_loss_bce(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE Loss with Binary Cross Entropy"""
        # Reconstruction loss
        recon_x = torch.clamp(recon_x, 1e-8, 1 - 1e-8)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        
        batch_size = x.size(0)
        return total_loss / batch_size, recon_loss / batch_size, kl_loss / batch_size
    
    def vae_loss_mse(self, recon_x, x, mu, logvar, beta=1.0):
        """VAE Loss with Mean Squared Error"""
        # Reconstruction loss
        recon_x_flat = recon_x.view(recon_x.size(0), -1)
        x_flat = x.view(x.size(0), -1)
        recon_loss = F.mse_loss(recon_x_flat, x_flat, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + beta * kl_loss
        
        batch_size = x.size(0)
        return total_loss / batch_size, recon_loss / batch_size, kl_loss / batch_size
    
    def denormalize_image(self, image):
        """反标准化图像"""
        image = image.clone()
        
        if self.config.use_bce:
            # BCE data range in [0,1]，clip to [0,1]
            return torch.clamp(image, 0, 1)
        else:
            # MSE data is normalized, need to denormalize
            # For Tanh output, convert from [-1,1] to [0,1]
            if isinstance(self.model.decoder[-1], nn.Tanh):
                image = (image + 1) / 2  # from [-1,1] to [0,1]
            
            # Apply denormalization
            image = image * self.std + self.mean
            return torch.clamp(image, 0, 1)
        
    def generate_and_save_samples(self, epoch=None, num_samples=16):
        """生成并保存样本"""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
            samples = self.denormalize_image(samples.cpu())
            
            # 创建图像网格
            ncols = 4
            nrows = (num_samples + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
            
            for i in range(num_samples):
                if nrows == 1:
                    ax = axes[i % ncols]
                else:
                    ax = axes[i // ncols, i % ncols]
                
                img = samples[i].numpy().transpose(1, 2, 0)
                if self.channels == 1:
                    ax.imshow(img[:, :, 0], cmap='gray')
                else:
                    ax.imshow(img)
                ax.axis('off')
            
            # Hide unused subplots
            for i in range(num_samples, nrows * ncols):
                if nrows == 1:
                    axes[i].axis('off')
                else:
                    axes[i // ncols, i % ncols].axis('off')
            
            if epoch:
                plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=14, fontweight='bold')
                save_path = self.config.sample_dir / f"generated_epoch_{epoch}.png"
            else:
                plt.suptitle('Generated Samples', fontsize=14, fontweight='bold')
                save_path = self.config.sample_dir / "generated_samples.png"
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Generated samples saved to: {save_path}")
            
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.view(data.size(0), -1).to(self.device)
            batch_size = data.size(0)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            recon_batch, mu, logvar = self.model(data)
            
            # 计算损失
            loss, recon_loss, kl_loss = self.loss_fn(recon_batch, data, mu, logvar)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item() / batch_size,
                'recon': recon_loss.item() / batch_size,
                'kl': kl_loss.item() / batch_size
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        
        return avg_loss, avg_recon, avg_kl
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.view(data.size(0), -1).to(self.device)
                recon_batch, mu, logvar = self.model(data)
                loss, recon_loss, kl_loss = self.loss_fn(recon_batch, data, mu, logvar)
                
                total_loss += loss.item()
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        avg_recon = total_recon / len(val_loader.dataset)
        avg_kl = total_kl / len(val_loader.dataset)
        
        return avg_loss, avg_recon, avg_kl
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"Starting training on device: {self.device}")
        print(f"Training parameters: epochs={self.config.epochs}, lr={self.config.learning_rate}")
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{self.config.epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss, train_recon, train_kl = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_recon, val_kl = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_recon'].append(train_recon)
            self.history['train_kl'].append(train_kl)
            self.history['val_loss'].append(val_loss)
            self.history['val_recon'].append(val_recon)
            self.history['val_kl'].append(val_kl)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print results
            print(f"\nTraining results:")
            print(f"  Total loss: {train_loss:.4f}")
            print(f"  Reconstruction loss: {train_recon:.4f}")
            print(f"  KL divergence: {train_kl:.4f}")
            print(f"\nValidation results:")
            print(f"  Total loss: {val_loss:.4f}")
            print(f"  Reconstruction loss: {val_recon:.4f}")
            print(f"  KL divergence: {val_kl:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model.pth")
                print(f"Saving best model (Validation loss: {val_loss:.4f})")
            
            # 定期保存检查点和生成样本
            if epoch % 10 == 0 or epoch == self.config.epochs or epoch == 1:
                self.save_model(f"model_epoch_{epoch}.pth")
                self.generate_and_save_samples(epoch, num_samples=16)
                self.plot_reconstruction_comparison(epoch, val_loader)
        
        # 训练完成后绘制训练曲线
        self.plot_training_curves()
        print(f"\nTraining complete! All results saved in: {self.config.base_dir}")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'latent_dim': self.config.latent_dim,   
                'hidden_dims': self.config.hidden_dims,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs
            }
        }
                
        save_path = self.config.model_dir / filename
        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")
    

    def plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # 1. 总损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='training loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='validation loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=10)
        
        # 2. 重构损失曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_recon'], 'b-', label='training reconstruction loss', linewidth=2)
        ax2.plot(epochs, self.history['val_recon'], 'r-', label='validation reconstruction loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Reconstruction Loss', fontsize=12)
        ax2.set_title('Reconstruction Loss Curve', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=10)
        
        # 3. KL散度曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['train_kl'], 'b-', label='training KL divergence', linewidth=2)
        ax3.plot(epochs, self.history['val_kl'], 'r-', label='validation KL divergence', linewidth=2)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('KL Divergence', fontsize=12)
        ax3.set_title('KL Divergence Curve', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(labelsize=10)
        
        # 4. 学习率曲线
        ax4 = axes[1, 1]
        ax4.plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        ax4.set_xlabel('Epoch', fontsize=12)
        ax4.set_ylabel('Learning Rate', fontsize=12)
        ax4.set_title('Learning Rate Curve', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(labelsize=10)
        ax4.set_yscale('log')
        
        plt.suptitle('VAE Training Visualization', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # 保存图像
        save_path = self.config.plot_dir / "training_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
        plt.close(fig)


    def plot_reconstruction_comparison(self, epoch, dataloader, num_images=4):
        """Plot comparison between original and reconstructed images"""
        if dataloader is None:
            print("No dataloader provided for reconstruction comparison")
            return
        
        self.model.eval()
        
        # Get a batch of data
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        images = images[:num_images]
        
        with torch.no_grad():
            # Reconstruct images
            recon_images, _, _ = self.model(images)
            recon_images = recon_images.cpu()
            images = images.cpu()
        
        # 反标准化
        orig_images = self.denormalize_image(images)
        recon_images = self.denormalize_image(recon_images)
            
        # Create comparison plot
        fig, axes = plt.subplots(2, num_images, figsize=(3*num_images, 6))
        
        for i in range(num_images):
            if self.channels == 1:
                # grayscale image
                axes[0, i].imshow(orig_images[i].squeeze().numpy(), 
                                cmap='gray', vmin=0, vmax=1)
            else:
                # RGB image - convert to HWC format and ensure values are in [0,1]
                img_np = orig_images[i].numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                axes[0, i].imshow(img_np)
            
            axes[0, i].set_title(f'Original {i+1}', fontsize=10)
            axes[0, i].axis('off')
            
            # 重建图像
            if self.channels == 1:
                axes[1, i].imshow(recon_images[i].squeeze().numpy(), 
                                cmap='gray', vmin=0, vmax=1)
            else:
                img_np = recon_images[i].numpy().transpose(1, 2, 0)
                img_np = np.clip(img_np, 0, 1)
                axes[1, i].imshow(img_np)
            
            axes[1, i].set_title(f'Reconstructed {i+1}', fontsize=10)
            axes[1, i].axis('off')
        
        # 添加标题
        if epoch:
            title = f'Reconstruction Comparison - Epoch {epoch}'
            save_path = self.config.plot_dir / f"reconstruction_epoch_{epoch}.png"
        else:
            title = 'Reconstruction Comparison'
            save_path = self.config.plot_dir / "reconstruction_comparison.png"
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save image
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Reconstruction comparison saved to: {save_path}")
        plt.close(fig)

def plot_latent_space(model, device, dataloader, config, num_samples=1000):
    """Visualize latent space (only if latent dimension >= 2)"""
    if model.fc_mu.out_features < 2:
        print("Latent dimension less than 2, cannot visualize latent space")
        return
    
    model.eval()
    all_mus = []
    all_labels = []
    
    with torch.no_grad():
        count = 0
        for data, labels in dataloader:
            if count >= num_samples:
                break
                
            data = data.view(data.size(0), -1).to(device)
            mu, _ = model.encode(data)
            
            all_mus.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())
            count += data.size(0)
    
    all_mus = np.concatenate(all_mus, axis=0)[:num_samples]
    all_labels = np.concatenate(all_labels, axis=0)[:num_samples]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(all_mus[:, 0], all_mus[:, 1], 
                        c=all_labels, cmap='tab10', alpha=0.6, s=10)
    
    plt.colorbar(scatter, label='Number Label')
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.title('Latent Space Distribution (First Two Dimensions)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    save_path = config.plot_dir / "latent_space_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Latent space distribution saved to: {save_path}")
    plt.close(scatter)


def load_model(config, filename, device):
    """Load model checkpoint"""
    load_path = config.model_dir / filename
    
    if not load_path.exists():
        print(f"Model file does not exist: {load_path}")
        return None
    
    checkpoint = torch.load(load_path, map_location=device)
    
    # Create model
    model = FlexibleVAE(
        input_dim=784,
        hidden_dim=checkpoint['config']['hidden_dim'],
        latent_dim=checkpoint['config']['latent_dim']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {load_path}")
    return model, checkpoint['optimizer_state_dict'], checkpoint['history']

# ==================== Main Function ====================
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='VAE Training')
    
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['mnist', 'cifar10', 'cifar100', 'celeba'],
                       help='Dataset to use')
    parser.add_argument('--img_size', type=int, nargs=2, default=None,
                       help='Image size (height width)')
    
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='潜在空间维度')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256],
                       help='隐藏层维度')
    parser.add_argument('--use_bce', action='store_true',
                       help='Use BCE loss (default is MSE)')
    
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    
    parser.add_argument('--resume', type=str, default=False, help='Resume training from checkpoint')
    parser.add_argument('--base_dir', type=str, default='/home/yinjx1/code/DCGAN/vae_expr2', help='Base directory for experiments')
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        base_dir=args.base_dir,
        exp_name=args.exp_name,
        dataset=args.dataset,
        img_size=tuple(args.img_size) if args.img_size else None,
        latent_dim=args.latent_dim,
        hidden_dims=args.hidden_dims,
        use_bce=args.use_bce,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    try:
        train_loader, val_loader, test_loader, dataset_info = DatasetManager.load_dataset(
            dataset_name=args.dataset,
            data_root='./data',
            batch_size=config.batch_size,
            img_size=config.img_size
        )
        
        # 更新配置中的图像尺寸
        if config.img_size is None:
            config.img_size = dataset_info['img_size']
        
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = FlexibleVAE(
        img_size=config.img_size,
        channels=dataset_info['channels'],
        hidden_dims=config.hidden_dims,
        latent_dim=config.latent_dim,
        use_bce=config.use_bce
    )
    
    # Compute and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (Trainable: {trainable_params:,})")
    
    # Create trainer
    trainer = FlexibleVAETrainer(model, device, config, dataset_info)
    
    # Check if resuming from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model, optimizer_state, history = load_model(config, args.resume, device)
        if model:
            trainer.model = model
            trainer.optimizer.load_state_dict(optimizer_state)
            trainer.history = history
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader)
    
     # 4. Visualize latent space (if latent dimension >= 2)
    if config.latent_dim >= 2:
        plot_latent_space(trainer.model, device, test_loader, config)
        
    print(f"\nAll results saved to: {config.base_dir}")
    print("Experiment completed!")
    
if __name__ == "__main__":
    main()