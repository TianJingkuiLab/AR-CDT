# AR-CDT Net

import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import sys
import math
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gmean

# =====================================================================================
# --- New DynamicTanh model ---
# =====================================================================================
class ChannelwiseDynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, channels_last=True):
        super(ChannelwiseDynamicTanh, self).__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last
        self.alpha = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
        return x
# =====================================================================================
# ---  SE merge ---
# =====================================================================================
class SqueezeExcite(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcite, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MSDConv(nn.Module):
    def __init__(self, inc, outc, kernel_sizes=[3, 5, 7], padding_modes='zeros'):
        super(MSDConv, self).__init__()
        # 尺度的数量由kernel_sizes列表的长度决定
        self.num_scales = len(kernel_sizes)
        self.inc = inc
        self.outc = outc

        # p_conv生成偏移量
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc * 2 * self.num_scales, 3, 1, 1, groups=inc),
            nn.BatchNorm2d(inc * 2 * self.num_scales),
            nn.ReLU(),
            nn.Conv2d(inc * 2 * self.num_scales, 2 * self.num_scales, 1)
        )
        
        # m_conv 和 b_conv 保持不变
        self.m_conv = nn.Sequential(nn.Conv2d(inc, outc, 3, 1, 1), nn.LeakyReLU(), nn.Conv2d(outc, outc, 3, 1, 1))
        self.b_conv = nn.Sequential(nn.Conv2d(inc, outc, 3, 1, 1), nn.LeakyReLU(), nn.Conv2d(outc, outc, 3, 1, 1))
        
        # 创建不同感受野大小的卷积层
        self.convs_per_scale = nn.ModuleList([
            nn.Conv2d(inc, outc, kernel_size=ks, stride=1, padding=(ks - 1) // 2, bias=False) for ks in kernel_sizes
        ])
        
        # SE融合不同尺度的信息
        self.fusion_attention = SqueezeExcite(outc * self.num_scales)
        self.final_conv = nn.Conv2d(outc * self.num_scales, outc, 1)
    
    def forward(self, x, epoch, hw_range):
        B, C, H, W = x.shape
        offsets = self.p_conv(x)
        scale_features = []
        
        for i in range(self.num_scales):
            offset_i = offsets[:, 2*i:2*(i+1), :, :]
            grid = F.affine_grid(torch.eye(2, 3, device=x.device).unsqueeze(0).expand(B, -1, -1), (B, C, H, W), align_corners=True)
            deformed_grid = grid + offset_i.permute(0, 2, 3, 1)
            
            # 每一条路径都使用自己的形变，并应用自己独特尺寸的卷积核
            scale_feature = F.grid_sample(x, deformed_grid, mode='bilinear', padding_mode='border', align_corners=True)
            scale_feature = self.convs_per_scale[i](scale_feature)
            scale_features.append(scale_feature)
            
        multi_scale_features = torch.cat(scale_features, dim=1)
        fused_features = self.fusion_attention(multi_scale_features)
        fused_features = self.final_conv(fused_features)
        
        m = torch.tanh(self.m_conv(x))
        b = self.b_conv(x)
        return fused_features * m + b

# =====================================================================================
# --- 主模型 ---
# =====================================================================================
class ARConvNet(nn.Module):
    def __init__(self, num_features, num_classes, bottleneck_dim, arconv_out_c, dropout_rate):
        super(ARConvNet, self).__init__()
        self.bottleneck_dim = bottleneck_dim
        self.epoch = 0
        
        self.bottleneck = nn.Linear(num_features, self.bottleneck_dim)

        self.spatial_expert = MSDConv(
            inc=1, 
            outc=arconv_out_c,
            kernel_sizes=[3, 5, 7]
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        
        final_dim = arconv_out_c
        classifier_hidden_dim = final_dim // 2
        
        # 分类器结构
        self.final_classifier = nn.Sequential(
            nn.Linear(final_dim, classifier_hidden_dim),
            ChannelwiseDynamicTanh(normalized_shape=classifier_hidden_dim, channels_last=True),
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_hidden_dim, num_classes)
        )

    def forward(self, x_orig):
        x_bottleneck = self.bottleneck(x_orig)
        img_side = int(math.sqrt(self.bottleneck_dim))
        x_image = x_bottleneck.view(-1, 1, img_side, img_side)
        
        spatial_features = self.spatial_expert(x_image, self.epoch, (img_side, img_side))
        
        final_features = self.pool(spatial_features).flatten(1)
        output = self.final_classifier(final_features)
        return F.log_softmax(output, dim=1)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction
    def forward(self, inputs, targets):
        log_pt = F.log_softmax(inputs, dim=1).gather(1, targets.view(-1, 1)).squeeze(1)
        pt = torch.exp(log_pt)
        focal_loss = -1 * (1 - pt)**self.gamma * log_pt
        if self.alpha is not None:
            if self.alpha.device != focal_loss.device: self.alpha = self.alpha.to(focal_loss.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def clr_transform(X_raw, pseudocount=1e-6):
    X_plus_pseudocount = np.array(X_raw, dtype=np.float64) + pseudocount
    geometric_means = gmean(X_plus_pseudocount, axis=1, keepdims=True)
    geometric_means[geometric_means == 0] = 1
    X_clr = np.log(X_plus_pseudocount / geometric_means)
    return np.nan_to_num(X_clr, nan=0.0, posinf=0.0, neginf=0.0)

def evaluate(model, X_data, y_data, criterion, device):
    model.eval()
    with torch.no_grad():
        outputs = model(X_data)
        loss = criterion(outputs, y_data) if criterion else None
        probs = torch.exp(outputs)
    return loss, probs

def main():
    parser = argparse.ArgumentParser(description="Final Model")
    parser.add_argument('--abundance_csv', type=str, required=True, help="丰度文件 (物种x样本)")
    parser.add_argument('--metadata_csv', type=str, required=True, help="元数据文件")
    parser.add_argument('--disease_label', type=int, default=None, help="指定单个疾病标签进行二分类，否则进行多分类")
    # --- 新增代码: 添加阈值参数 ---
    parser.add_argument('--threshold', type=float, default=0.03, help="物种丰度过滤阈值")
    parser.add_argument('--bottleneck_dim', type=int, default=64, help="降维后的特征维度")
    parser.add_argument('--arconv_out_c', type=int, default=64, help="ARConv路径的输出通道数")
    parser.add_argument('--n_splits', type=int, default=10, help="交叉验证的折数")
    parser.add_argument('--n_repeats', type=int, default=1, help="交叉验证的重复次数")
    parser.add_argument('--random_seed', type=int, default=42, help="全局随机种子")
    parser.add_argument('--epochs', type=int, default=300, help="最大训练轮数")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="权重衰减")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout比率")
    parser.add_argument('--patience', type=int, default=40, help="早停的耐心轮数")
    parser.add_argument('--gamma', type=float, default=2.0, help="Focal Loss的聚焦参数gamma")
    
    args = parser.parse_args()
    
    img_side = int(math.sqrt(args.bottleneck_dim))
    if img_side * img_side != args.bottleneck_dim:
        sys.exit(f"错误: bottleneck_dim ({args.bottleneck_dim}) 必须是一个可以完美开方的数字。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    print("--- 1. 正在加载和准备数据... ---")
    df_abundance_raw = pd.read_csv(args.abundance_csv, index_col=0)
    df_abundance = df_abundance_raw.transpose()

    if df_abundance.sum(axis=1).median() > 90: # 
        print("--- 检测到丰度数据为百分比格式，正在转换为比例 (除以100)... ---")
        df_abundance = df_abundance / 100.0
    
    # --- 新增代码: 应用物种丰度过滤 ---
    if args.threshold > 0:
        print(f"--- 正在应用物种丰度过滤 (阈值: {args.threshold}) ---")
        original_num_species = df_abundance.shape[1]
        # 计算每个物种(列)的总丰度
        sum_abundance_per_species = df_abundance.sum(axis=0) 
        # 创建一个布尔掩码来保留物种
        species_to_keep_mask = sum_abundance_per_species >= args.threshold
        # 应用掩码过滤数据
        df_abundance = df_abundance.loc[:, species_to_keep_mask]
        print(f"原始物种数量: {original_num_species}, 过滤后: {df_abundance.shape[1]}")
        print(f"-----------------------------------")


    df_metadata = pd.read_csv(args.metadata_csv)
    
    if args.disease_label is not None:
        print(f"--- 模式: 二分类 (疾病标签: {args.disease_label} vs 健康: 0) ---")
        healthy_label = 0
        df_metadata = df_metadata[df_metadata['Group'].isin([healthy_label, args.disease_label])]
    else:
        print("--- 模式: 多分类 ---")

    common_samples = df_abundance.index.intersection(df_metadata['sampleID'])
    if len(common_samples) == 0: sys.exit("\n【致命错误】: 找不到任何共同的样本ID！请检查文件。")
    
    print(f"成功找到 {len(common_samples)} 个共同样本。")
    X_raw = df_abundance.loc[common_samples].values
    y_raw = df_metadata.set_index('sampleID').loc[common_samples]['Group'].values
    
    print("--- 2. 正在对数据进行CLR变换... ---")
    X_processed = clr_transform(X_raw)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    num_features, num_classes = X_processed.shape[1], len(le.classes_)
    
    print(f"数据准备完毕: {X_processed.shape[0]}个样本, {num_features}个特征, {num_classes}个类别。")
    print(f"--- 3. 开始执行 {args.n_repeats}次重复 x {args.n_splits}折 交叉验证 ---")
    kf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_seed)
    
    fold_aucs, fold_accs, fold_f1s, fold_recalls = [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_processed, y_encoded)):
        print(f"\n--- 正在处理: 第 {fold_idx+1}/{args.n_splits * args.n_repeats} 折 ---")
        
        X_train, X_val = X_processed[train_idx], X_processed[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
        
        X_train_tensor = torch.tensor(X_train, dtype=torch.float).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        
        model = ARConvNet(
            num_features=num_features, 
            num_classes=num_classes, 
            bottleneck_dim=args.bottleneck_dim, 
            arconv_out_c=args.arconv_out_c,
            dropout_rate=args.dropout_rate
        ).to(device)
        
        num_samples_per_class = torch.bincount(y_train_tensor).float()
        alpha_weights = 1.0 / (num_samples_per_class + 1e-9)
        alpha_weights = alpha_weights / alpha_weights.sum()
        criterion = FocalLoss(alpha=alpha_weights, gamma=args.gamma)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_val_auc, epochs_no_improve = 0.0, 0
        best_acc, best_f1, best_recall = 0.0, 0.0, 0.0
        
        for epoch in range(args.epochs):
            model.train()
            model.epoch = epoch
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            val_loss, val_probs = evaluate(model, X_val_tensor, y_val_tensor, criterion, device)
            val_preds = torch.argmax(val_probs, dim=1).cpu().numpy()
            val_true = y_val_tensor.cpu().numpy()
            if num_classes == 2: val_auc = roc_auc_score(val_true, val_probs.cpu()[:, 1])
            else: val_auc = roc_auc_score(val_true, val_probs.cpu(), multi_class='ovr')
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_acc = accuracy_score(val_true, val_preds)
                best_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
                best_recall = recall_score(val_true, val_preds, average='macro', zero_division=0)
                epochs_no_improve = 0
            else: epochs_no_improve += 1
            if epochs_no_improve >= args.patience: break
        
        print(f"此折最佳结果: Val AUC: {best_val_auc:.4f}, F1: {best_f1:.4f}, Recall: {best_recall:.4f}")
        fold_aucs.append(best_val_auc)
        fold_accs.append(best_acc)
        fold_f1s.append(best_f1)
        fold_recalls.append(best_recall)
        
    print(f"\n--- 4. 交叉验证最终结果 ---")
    print(f"平均验证 AUC:     {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"平均验证 ACC:     {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"平均验证 F1-Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"平均验证 Recall:  {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")


if __name__ == '__main__':  
    main()