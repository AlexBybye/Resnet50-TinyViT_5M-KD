import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.teacher import get_teacher_model
from models.student import TinyViTStudent
from data.transforms import get_train_transform, get_val_transform
from utils.logger import Logger
from utils.scheduler import get_scheduler


class DistillationTrainer:
    def __init__(self, config_path):
        # 加载配置
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        # 初始化组件
        self._init_data()
        self._init_models()
        self._init_optim()
        self.logger = Logger()

    def _init_data(self):
        train_transform = get_train_transform(self.cfg['data']['img_size'])
        val_transform = get_val_transform(self.cfg['data']['img_size'])

        # 自定义数据集类需要实现
        self.train_set = CustomDataset(
            self.cfg['data']['train_dir'],
            transform=train_transform
        )
        self.val_set = CustomDataset(
            self.cfg['data']['val_dir'],
            transform=val_transform
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=self.cfg['train']['batch_size'],
            shuffle=True,
            num_workers=self.cfg['data']['num_workers']
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=self.cfg['train']['batch_size'],
            shuffle=False
        )

    def _init_models(self):
        # 教师模型
        self.teacher = get_teacher_model(self.cfg)
        self.teacher.eval()

        # 学生模型
        self.student = TinyViTStudent(self.cfg)
        self.student.to(self.cfg['train']['device'])

    def _init_optim(self):
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg['train']['lr']
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.cfg['distill']['scheduler']
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.cfg['train']['amp']
        )

    def _compute_loss(self, teacher_out, student_out, labels, tau):
        # 知识蒸馏损失
        soft_target = F.softmax(teacher_out['logits'] / tau, dim=1)
        log_prob = F.log_softmax(student_out['logits'] / tau, dim=1)
        kd_loss = F.kl_div(log_prob, soft_target, reduction='batchmean') * (tau ** 2)

        # 特征对齐损失
        feat_loss = sum(
            F.mse_loss(s_feat, t_feat)
            for s_feat, t_feat in zip(
                student_out['features'],
                teacher_out['features']
            )
        )

        # 任务损失
        task_loss = F.cross_entropy(student_out['logits'], labels)

        return (
                self.cfg['distill']['loss_weights']['task'] * task_loss +
                self.cfg['distill']['loss_weights']['kd'] * kd_loss +
                self.cfg['distill']['loss_weights']['feature'] * feat_loss
        )

    def _train_epoch(self, epoch):
        self.student.train()
        total_loss = 0.0

        # 动态温度计算
        tau = self.cfg['distill']['temperature']['max'] * (
                self.cfg['distill']['temperature']['decay'] ** epoch
        )
        tau = max(tau, self.cfg['distill']['temperature']['min'])

        for inputs, labels in tqdm(self.train_loader):
            inputs = inputs.to(self.cfg['train']['device'])
            labels = labels.to(self.cfg['train']['device'])

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.cfg['train']['amp']):
                # 教师推理
                with torch.no_grad():
                    teacher_out = self.teacher(inputs)

                # 学生推理
                student_out = self.student(inputs)

                # 计算损失
                loss = self._compute_loss(
                    teacher_out, student_out, labels, tau)

            # 反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def _validate(self):
        self.student.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.cfg['train']['device'])
                outputs = self.student(inputs)['logits']
                preds = outputs.argmax(dim=1)
                correct += (preds == labels.to(self.cfg['train']['device'])).sum().item()
                total += labels.size(0)

        return correct / total

    def train(self):
        best_acc = 0.0
        patience_counter = 0

        for epoch in range(self.cfg['train']['epochs']):
            # 训练阶段
            train_loss = self._train_epoch(epoch)

            # 验证阶段
            val_acc = self._validate()

            # 学习率调整
            self.scheduler.step(val_acc)

            # 记录日志
            self.logger.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })

            # 早停判断
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.student.state_dict(), "best_student.pth")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg['distill']['early_stop_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


if __name__ == "__main__":
    trainer = DistillationTrainer("configs/distill.yaml")
    trainer.train()