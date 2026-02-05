import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import matplotlib
# 使用非交互后端
matplotlib.use('Agg')  # 必须在导入pyplot之前
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from datetime import datetime

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 配置和路径设置 ====================
class Config:
    def __init__(self, base_dir=None):
        self.batch_size = 128
        self.epochs = 50
        self.latent_dim = 20
        self.hidden_dim = 400
        self.learning_rate = 1e-3
        self.base_dir = base_dir
        
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

# ==================== 数据准备 ====================
def get_mnist_dataloaders(batch_size=128):
    """获取MNIST数据加载器"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader

# ==================== VAE模型 ====================
class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, dropout_rate=0.2):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出在0-1之间
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ==================== 损失函数 ====================
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数
    beta: β-VAE参数，控制KL散度权重
    """
    # 重构损失 - 使用BCE
    # 注意：需要将数据从标准化范围转换回[0,1]
    recon_x = torch.clamp(recon_x, 1e-8, 1 - 1e-8)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失
    total_loss = BCE + beta * KLD
    
    return total_loss, BCE, KLD

# ==================== 训练器 ====================
class VAETrainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # 记录训练历史
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'learning_rates': []
        }
        
        plt.ioff()  # 关闭交互模式
    
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
            loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
            
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
                loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
                
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
        """保存模型检查点"""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'latent_dim': self.config.latent_dim,   
                'hidden_dim': self.config.hidden_dim,
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs
            }
        }
                
        save_path = self.config.model_dir / filename
        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")
    

    def plot_training_curves(self):
        """绘制训练曲线"""
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

    def generate_and_save_samples(self, epoch, num_samples=16):
        """Generate and save sample images"""
        self.model.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(num_samples, self.model.fc_mu.out_features).to(self.device)
            samples = self.model.decode(z).cpu()
            
            # Reshape to images
            samples = samples.view(-1, 1, 28, 28)
            
            # Create image grid
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            
            for i, ax in enumerate(axes.flat):
                # Denormalize
                img = samples[i].squeeze().numpy()
                img = img * 0.3081 + 0.1307  # Denormalize
                img = np.clip(img, 0, 1)  # Ensure within [0,1]
                
                ax.imshow(img, cmap='gray')
                ax.axis('off')
            
          
            plt.suptitle(f'VAE Generated Samples - Epoch {epoch}', fontsize=16, fontweight='bold')
            save_path = self.config.sample_dir / f"generated_samples_epoch_{epoch}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Generated samples saved to: {save_path}")
            plt.close(fig)

    def plot_reconstruction_comparison(self, epoch, dataloader, num_images=4):
        """Plot comparison between original and reconstructed images"""
        self.model.eval()
        
        # Get a batch of data
        data_iter = iter(dataloader)
        images, labels = next(data_iter)
        images = images[:num_images]
        
        with torch.no_grad():
            # Reconstruct images
            images_flat = images.view(num_images, -1).to(self.device)
            recon_images, _, _ = self.model(images_flat)
            recon_images = recon_images.view(-1, 1, 28, 28).cpu()
        
        # Create comparison plot
        fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
        
        for i in range(num_images):
            # Original images (denormalize)
            orig_img = images[i].squeeze().numpy()
            orig_img = orig_img * 0.3081 + 0.1307  # 反标准化
            orig_img = np.clip(orig_img, 0, 1)
            
            # Reconstructed images (denormalize)
            recon_img = recon_images[i].squeeze().numpy()
            recon_img = recon_img * 0.3081 + 0.1307  # Denormalize
            recon_img = np.clip(recon_img, 0, 1)
            
            # Plot original images
            axes[0, i].imshow(orig_img, cmap='gray')
            axes[0, i].set_title(f'Label: {labels[i].item()}', fontsize=10)
            axes[0, i].axis('off')
            
            # Plot reconstructed images
            axes[1, i].imshow(recon_img, cmap='gray')
            axes[1, i].set_title('Reconstructed', fontsize=10)
            axes[1, i].axis('off')
        
        plt.suptitle('Original vs Reconstructed Images', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save image
        save_path = self.config.plot_dir / f"reconstruction_comparison_epoch_{epoch}.png"
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
    model = VAE(
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
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=20, help='Latent space dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--resume', type=str, default=False, help='Resume training from checkpoint')
    parser.add_argument('--base_dir', type=str, default='/home/yinjx1/code/DCGAN/vae_expr', help='Base directory for experiments')
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(Path(args.base_dir))
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.latent_dim = args.latent_dim
    config.learning_rate = args.learning_rate

    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(config.batch_size)
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = VAE(
        input_dim=784,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim
    )
    
    # Create trainer
    trainer = VAETrainer(model, device, config)
    
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