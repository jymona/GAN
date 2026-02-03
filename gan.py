#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import datetime
from pathlib import Path
import shutil

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 定义生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, img_size=28):
        super(Generator, self).__init__()
        
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# 2. 定义判别器
class Discriminator(nn.Module):
    def __init__(self, channels=1, img_size=28):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        
        # 计算经过4次下采样后的特征图尺寸
        # 每次下采样：stride=2，所以尺寸减半
        self.ds_size = img_size // (2 ** 4)  # 28 -> 14 -> 7 -> 3 -> 1 (对于28x28图像)
        
        self.model = nn.Sequential(
            # 第1层: channels -> 16
            # 输入: [batch_size, channels, img_size, img_size]
            nn.Conv2d(
                in_channels=channels, 
                out_channels=16, 
                kernel_size=3, 
                stride=2, 
                padding=1),  # 16 x 14 x 14
            # 输出: [batch_size, 16, img_size//2, img_size//2]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 第2层: 16 -> 32
            # 输入: [batch_size, 16, img_size//2, img_size//2]
            # 输出: [batch_size, 32, img_size//4, img_size//4]
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32 x 7 x 7
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 第3层: 32 -> 64
            # 输入: [batch_size, 32, img_size//4, img_size//4]
            # 输出: [batch_size, 64, img_size//8, img_size//8]
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64 x 3 x 3
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            # 第4层: 64 -> 128
            # 输入: [batch_size, 64, img_size//8, img_size//8]
            # 输出: [batch_size, 128, img_size//16, img_size//16]
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 128 x 1 x 1 (对于28x28)
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )  
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
            
        # def discriminator_block(in_filters, out_filters, bn=True):
        #     block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, 0.8))
        #     block.extend([
        #         nn.LeakyReLU(0.2, inplace=True),
        #         nn.Dropout2d(0.25)
        #     ])
        #     return block
        
        # self.model = nn.Sequential(
        #     *discriminator_block(channels, 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128)
        # )
        
        # 计算全连接层输入维度
        # 对于28x28图像: 128 * 1 * 1 = 128
        # 对于32x32图像: 128 * 2 * 2 = 512
        # self.fc_input_dim = 128 * self.ds_size * self.ds_size
        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_input_dim, 1),
        #     nn.Sigmoid()
        # )
    
    def forward(self, img):
        features = self.model(img)
        pooled = self.adaptive_pool(features)  # [batch, 128, 1, 1]
        flattened = pooled.view(pooled.size(0), -1)  # [batch, 128]
        validity = self.fc(flattened)
        return validity

# 3. 数据加载和预处理
class DatasetManager:
    def __init__(self, dataset_name='mnist', batch_size=64):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def download_dataset(self):
        """下载数据集"""
        if self.dataset_name == 'mnist':
            # MNIST数据集
            train_dataset = datasets.MNIST(
                root='./data', 
                train=True,
                download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'fashion-mnist':
            # Fashion-MNIST数据集
            train_dataset = datasets.FashionMNIST(
                root='./data',
                train=True,
                download=True,
                transform=self.transform
            )
        elif self.dataset_name == 'cifar10':
            # CIFAR-10数据集
            self.transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=self.transform
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return train_dataset
    
    def get_dataloader(self):
        """获取数据加载器"""
        dataset = self.download_dataset()
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        return dataloader

# 4. GAN训练器
class GANTrainer:
    def __init__(self, dataset_name='mnist', latent_dim=100, lr=0.0002, d_train_multiple=5, save_dir='experiments'):
        self.dataset_name = dataset_name
        self.latent_dim = latent_dim
        self.lr = lr
        self.d_train_multiple = d_train_multiple
        self.save_dir = save_dir
        
        # 初始化模型
        self.generator = Generator(latent_dim=latent_dim).to(device)
        self.discriminator = Discriminator().to(device)
        
        # 损失函数
        self.adversarial_loss = nn.BCELoss()
        
        # 优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=lr, 
            betas=(0.5, 0.999)
        )
        
        # 创建保存目录 - 使用时间戳防止覆盖
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"{dataset_name}_{timestamp}"
        self.images_dir = self.save_dir / "images"
        self.checkpoints_dir = self.save_dir / "checkpoints"
        self.logs_dir = self.save_dir / "logs"
        
        for dir_path in [self.images_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"实验保存目录: {self.save_dir}")
        
        # 记录训练历史
        self.history = {
            'g_losses': [],
            'd_losses': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'd_train_steps': [],
            'g_train_steps': []
        }
        
        self.d_step = 0
        self.g_step = 0
    
    def train_epoch(self, dataloader, epoch, n_epochs):
        """训练一个epoch"""
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_d_real_acc = []
        epoch_d_fake_acc = []
        
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.shape[0] # [64, 1, 28, 28] or [64, 3, 32, 32]
            
            # 真实图像和标签
            real_imgs = imgs.to(device)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # ====================
            #  训练判别器
            # ====================
            d_loss_accum = 0
            d_real_acc_accum = 0
            d_fake_acc_accum = 0
            
            for d_step in range(self.d_train_multiple):
                self.optimizer_D.zero_grad()
                
                # 真实图像的损失
                real_pred = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_pred, valid)
                d_real_accuracy = real_pred.mean().item()
                
                # 生成假图像
                z = torch.randn(batch_size, self.latent_dim, device=device) # [64, 100]
                gen_imgs = self.generator(z) # 生成器生成图像，包含计算图
            
                # 假图像的损失
                fake_pred = self.discriminator(gen_imgs.detach()) # 使用detach()
                d_fake_loss = self.adversarial_loss(fake_pred, fake)
                d_fake_accuracy = 1 - fake_pred.mean().item()
            
                # 反向传播（只更新判别器）
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                if hasattr(self, 'use_graident_penalty') and self.use_graident_penalty:
                    d_loss += self.calculate_gradient_penalty(real_imgs, gen_imgs)
                                    
                d_loss.backward()
                self.optimizer_D.step()
                
                d_loss_accum += d_loss.item()
                d_real_acc_accum += d_real_accuracy
                d_fake_acc_accum += d_fake_accuracy
                
                self.d_step += 1
            
            # 平均D的训练结果
            avg_d_loss = d_loss_accum / self.d_train_multiple
            avg_d_real_acc = d_real_acc_accum / self.d_train_multiple
            avg_d_fake_acc = d_fake_acc_accum / self.d_train_multiple
            
            
            # ====================
            #  训练生成器
            # ====================
            self.optimizer_G.zero_grad()
            
            gen_imgs = self.generator(z)
            g_pred = self.discriminator(gen_imgs)
            g_loss = self.adversarial_loss(g_pred, valid)
            
            if hasattr(self, 'use_feature_matching') and self.use_feature_matching:
                real_features = self.get_discriminator_features(real_imgs)
                fake_features = self.get_discriminator_features(gen_imgs)
                feature_loss = torch.mean(torch.abs(real_features - fake_features))
                g_loss = g_loss + 0.1 * feature_loss    
            
            g_loss.backward()
            self.optimizer_G.step()
            
            self.g_step += 1
            
            # 记录损失
            epoch_g_losses.append(g_loss.item())
            epoch_d_losses.append(avg_d_loss)
            epoch_d_real_acc.append(avg_d_real_acc)
            epoch_d_fake_acc.append(avg_d_fake_acc)
            
            # 记录训练步数
            self.history['d_train_steps'].append(self.d_step)
            self.history['g_train_steps'].append(self.g_step)
            
            # 打印训练信息
            if i % 100 == 0:
                 print(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{len(dataloader)}] "
                      f"[D loss: {avg_d_loss:.4f}] [G loss: {g_loss.item():.4f}] "
                      f"[D-Real: {avg_d_real_acc:.3f}] [D-Fake: {avg_d_fake_acc:.3f}] "
                      f"[D-steps: {self.d_step}, G-steps: {self.g_step}]")
        
        # 更新历史记录
        self.history['g_losses'].append(np.mean(epoch_g_losses))
        self.history['d_losses'].append(np.mean(epoch_d_losses))
        self.history['d_real_acc'].append(np.mean(epoch_d_real_acc))
        self.history['d_fake_acc'].append(np.mean(epoch_d_fake_acc))
        
        return np.mean(epoch_g_losses), np.mean(epoch_d_losses)
    
    def calculate_gradient_penalty(self, real_imgs, fake_imgs):
        """计算WGAN-GP梯度惩罚"""
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=device)
        interpolates = (alpha * real_imgs + ((1 - alpha) * fake_imgs)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
        
    def get_discriminator_features(self, imgs):
        features = self.discriminator.model(imgs)
        features = self.discriminator.adaptive_pool(features)
        return features
    
    def sample_images(self, epoch, n_samples=25):
        """生成并保存样本图像"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        gen_imgs = self.generator(z)
        
        # 调整图像格式
        gen_imgs = 0.5 * gen_imgs + 0.5  # 反归一化
        
        # 创建图像网格
        fig, axes = plt.subplots(5, 5, figsize=(10, 10))
        for i, ax in enumerate(axes.flat):
            img = gen_imgs[i].cpu().detach().numpy()
            if img.shape[0] == 1:  # 灰度图
                img = img.squeeze()
                ax.imshow(img, cmap='gray')
            else:  # 彩色图
                img = img.transpose(1, 2, 0)
                ax.imshow(img)
            ax.axis('off')
        
        fig.suptitle(f'Epoch {epoch}', fontsize=16)
        plt.tight_layout()
        image_path = os.path.join(self.images_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(image_path)
        plt.close()
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        try: 
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'd_steps': self.d_step,
                'g_steps': self.g_step,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'history': self.history,
                'config': {
                    'latent_dim': self.latent_dim,
                    'd_train_multiple': self.d_train_multiple,
                    'dataset_name': self.dataset_name
                }
            }
            
            # 1. 先保存到临时文件
            temp_path = self.checkpoints_dir / f'temp_checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, temp_path)
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                final_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'
                os.rename(temp_path, final_path)  # 原子操作替换文件
                print(f"Checkpoint successfully saved at epoch {epoch}")
            else:
                print(f"Error: Temporary checkpoint file is invalid at epoch {epoch}")
                
            if is_best:
                best_path = self.checkpoints_dir / 'best_model.pth'
                # 直接复制最佳检查点
                shutil.copy2(final_path, best_path)
                print(f"✓ Best model saved to: {best_path}")
            
        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.history = checkpoint['history']
        self.d_step = checkpoint.get('d_steps', 0)
        self.g_step = checkpoint.get('g_steps', 0)
        return checkpoint['epoch']
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        epochs = range(1, len(self.history['g_losses']) + 1)
        
        # 1. 损失曲线
        axes[0, 0].plot(epochs, self.history['d_losses'], label='Discriminator Loss')
        axes[0, 0].plot(epochs, self.history['g_losses'], label='Generator Loss')
        axes[0, 0].set_title('Generator and Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 2. 判别器准确率
        axes[0, 1].plot(epochs, self.history['d_real_acc'], label='Real Accuracy')
        axes[0, 1].plot(epochs, self.history['d_fake_acc'], label='Fake Accuracy')
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_ylim(0, 1)
        
        # 3. 训练步数对比
        if len(self.history['d_train_steps']) > 0:
            batch_indices = range(len(self.history['d_train_steps']))
            axes[1, 0].plot(batch_indices, self.history['d_train_steps'], label='D steps')
            axes[1, 0].plot(batch_indices, self.history['g_train_steps'], label='G steps')
            axes[1, 0].set_title('Training Steps (D vs G)')
            axes[1, 0].set_xlabel('Batch Index')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 4. 损失比值
        loss_ratio = np.array(self.history['d_losses']) / (np.array(self.history['g_losses']) + 1e-8)
        axes[1, 1].plot(epochs, loss_ratio)
        axes[1, 1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('D Loss / G Loss Ratio')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Ratio')
        axes[1, 1].grid(True)
        
        # 5. 生成器损失
        axes[2, 0].plot(epochs, self.history['g_losses'], color='orange')
        axes[2, 0].set_title('Generator Loss')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Loss')
        axes[2, 0].grid(True)
        
        # 6. 判别器损失
        axes[2, 1].plot(epochs, self.history['d_losses'], color='blue')
        axes[2, 1].set_title('Discriminator Loss')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        history_path = self.save_dir / 'training_history.png'
        plt.savefig(history_path)
        plt.show()
        print(f"Training history saved to: {history_path}")

# 5. 主训练函数
def main():
    parser = argparse.ArgumentParser(description='Improved GAN Training')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion-mnist', 'cifar10'],
                       help='数据集选择')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='学习率')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='潜在空间维度')
    parser.add_argument('--d_train_multiple', type=int, default=5,
                       help='每训练1次G，训练D的次数')
    parser.add_argument('--save_dir', type=str, default='/home/yinjx1/code/DCGAN/experiments',
                       help='保存目录')
    parser.add_argument('--sample_interval', type=int, default=5,
                       help='保存样本图像的间隔')
    parser.add_argument('--save_interval', type=int, default=10,
                       help='保存检查点的间隔')
    
    args = parser.parse_args()
    
    # 训练配置
    config = {
        'dataset_name': args.dataset,
        'latent_dim': args.latent_dim,
        'batch_size': args.batch_size,
        'n_epochs': args.epochs,
        'lr': args.lr,
        'd_train_multiple': args.d_train_multiple,
        'sample_interval': args.sample_interval,
        'save_interval': args.save_interval,
        'save_dir': args.save_dir
    }
    
    print("GAN Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # 初始化数据加载器
    print("Loading dataset...")
    dataset_manager = DatasetManager(
        dataset_name=config['dataset_name'],
        batch_size=config['batch_size']
    )
    dataloader = dataset_manager.get_dataloader()
    print(f"Dataset loaded. Batch size: {config['batch_size']}, "
          f"Total batches: {len(dataloader)}")
    
    # 初始化训练器
    print("Initializing GAN trainer...")
    trainer = GANTrainer(
        dataset_name=args.dataset,
        latent_dim=args.latent_dim,
        lr=args.lr,
        d_train_multiple=args.d_train_multiple,
        save_dir=args.save_dir
    )
    
    # 训练循环
    print("Starting training...")
    best_g_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print('='*60)
        
        # 训练一个epoch
        g_loss, d_loss = trainer.train_epoch(dataloader, epoch, args.epochs)
        
        # 生成样本
        if epoch % args.sample_interval == 0 or epoch == 1:
            trainer.sample_images(epoch)
            print(f"Sample images saved for epoch {epoch}")
        
        # 保存检查点
        if epoch % args.save_interval == 0 or epoch == 1:
            is_best = g_loss < best_g_loss
            if is_best:
                best_g_loss = g_loss
            trainer.save_checkpoint(epoch, is_best=is_best)
            print(f"Checkpoint saved for epoch {epoch}")
            
        if epoch > 20 and g_loss > 3.0:  # 如果生成器损失太大
            print("Warning: Generator loss too high, consider adjusting learning rate")
    
    # 保存最终模型
    trainer.save_checkpoint(args.epochs, is_best=False)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    print("\nTraining completed!")
    print(f"Model saved in 'checkpoints/' directory")
    print(f"Generated images saved in 'images/' directory")

# 6. 推理演示
def inference_demo(checkpoint_path=None, latent_dim=100):
    """推理演示：生成新的图像"""
    # 加载训练好的模型
    generator = Generator(latent_dim=latent_dim).to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Model loaded from {checkpoint_path}")
    else:
        print("Using randomly initialized generator")
    
    # 生成图像
    generator.eval()
    
    with torch.no_grad():
        # 生成单个图像
        z = torch.randn(1, latent_dim, device=device)
        single_img = generator(z)
        
        # 生成多个图像
        z_grid = torch.randn(16, latent_dim, device=device)
        grid_imgs = generator(z_grid)
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 单个图像
    single_img = 0.5 * single_img.cpu().squeeze() + 0.5
    axes[0].imshow(single_img, cmap='gray')
    axes[0].set_title('Single Generated Image')
    axes[0].axis('off')
    
    # 图像网格
    grid = grid_imgs.cpu()
    grid = 0.5 * grid + 0.5
    grid = grid.view(4, 4, 28, 28)
    grid = grid.permute(0, 2, 1, 3).contiguous().view(4 * 28, 4 * 28)
    axes[1].imshow(grid, cmap='gray')
    axes[1].set_title('4x4 Grid of Generated Images')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('generated_samples.png')
    plt.show()

# 7. 运行程序
if __name__ == "__main__":
    main()
   