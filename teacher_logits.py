import torch
from tqdm import tqdm
from models.teacher import train_dataset  # 确保导入正确的预处理
from models.teacher import teacher_model, device, batch_size, num_workers


def save_teacher_logits():
    # 检查数据集有效性
    if len(train_dataset) == 0:
        raise ValueError("训练数据集为空，请检查数据路径和过滤逻辑")

    # 加载模型并移动到设备
    teacher_model.load_state_dict(torch.load('86best_teacher_model.pth'))  # 确认文件名正确
    teacher_model.to(device)
    teacher_model.eval()

    # 加载并冻结教师模型
    teacher_model.load_state_dict(torch.load('86best_teacher_model.pth'))

    # 关键修正：冻结所有参数
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model.to(device)
    teacher_model.eval()

    # 创建数据加载器（确保 shuffle=False 保持顺序）
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # 保存 logits 和标签
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="生成Logits"):
            images = images.to(device)
            logits = teacher_model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    # 合并数据并保存
    logits_tensor = torch.cat(all_logits)
    labels_tensor = torch.cat(all_labels)

    torch.save({
        'logits': logits_tensor,
        'labels': labels_tensor,
        'class_to_idx': train_dataset.class_to_idx  # 可选：保存类别映射
    }, 'teacher_logits.pt')

    print("Logits 和标签已保存至 teacher_logits.pt")


if __name__ == '__main__':
    save_teacher_logits()