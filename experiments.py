"""
实验脚本：对比 fit_uni 和 fit_unilasso 的训练效果

实验1：误差图
- 纵轴：平均 MSE
- 横轴：-log(lambda)
- 两条线：训练集误差、测试集误差
- 数据分割：三七分（70%训练，30%测试）
- 数据稀疏度：k-稀疏（只有 k 个特征有非零系数）
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, Optional
import warnings

# 导入 unilasso 相关函数
from unilasso import fit_unilasso, simulate_gaussian_data
from unilasso.uni_lasso import fit_uni


def simulate_sparse_gaussian_data(
    n: int = 1000,
    p: int = 20,
    sparsity: int = 5,
    signal_strength: float = 2.0,
    x_mean: float = 0,
    x_sd: float = 1,
    ysd: float = 1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成 k-稀疏的高斯线性回归数据。
    
    模型: y = X @ beta + epsilon
    其中 beta 只有 sparsity 个非零元素，其余为 0。
    
    Args:
        n: 样本数量
        p: 特征数量
        sparsity: 非零系数的数量 (k)
        signal_strength: 非零系数的大小（绝对值）
        x_mean: 特征的均值
        x_sd: 特征的标准差
        ysd: 噪声的标准差
        seed: 随机种子
        
    Returns:
        X: 特征矩阵 (n, p)
        y: 响应变量 (n,)
        beta_true: 真实的系数向量 (p,) - 用于评估模型性能
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成特征矩阵
    X = np.random.normal(size=(n, p), loc=x_mean, scale=x_sd)
    
    # 生成稀疏系数向量：随机选择 sparsity 个位置设置为非零
    beta_true = np.zeros(p)
    
    # 随机选择 sparsity 个特征作为有效特征
    nonzero_indices = np.random.choice(p, size=sparsity, replace=False)
    
    # 非零系数从 [-signal_strength, signal_strength] 均匀分布
    beta_true[nonzero_indices] = np.random.uniform(
        -signal_strength, signal_strength, size=sparsity
    )
    
    # 生成响应变量
    y = X @ beta_true + np.random.normal(size=n, scale=ysd)
    
    return X, y, beta_true


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def predict_with_path(result, X: np.ndarray) -> np.ndarray:
    """
    对给定数据 X，使用整个正则化路径上的所有模型进行预测。
    
    Args:
        result: fit_uni 或 fit_unilasso 的返回结果
        X: 特征矩阵
        
    Returns:
        预测结果矩阵，形状为 (n_samples, n_lambdas)
    """
    # 确保 X 是二维数组
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # 获取系数和截距
    coefs = result.coefs  # 形状: (n_lambdas, n_features) 或 (n_features,)
    intercepts = result.intercept  # 形状: (n_lambdas,) 或 标量
    
    # 处理一维情况
    if coefs.ndim == 1:
        coefs = coefs.reshape(1, -1)
        intercepts = np.array([intercepts])
    
    # 计算预测值: X @ coefs.T + intercepts
    # 结果形状: (n_samples, n_lambdas)
    predictions = X @ coefs.T + intercepts
    
    return predictions


def experiment_train_test_error(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = "unilasso",
    family: str = "gaussian",
    n_lmdas: int = 100,
    lmda_min_ratio: float = 1e-4,
    negative_penalty: float = 10.0,
    verbose: bool = False
) -> Dict:
    """
    实验1：计算模型在整个正则化路径上的训练集和测试集 MSE。
    
    Args:
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        model_type: 模型类型，"unilasso" 或 "uni"
        family: 分布家族
        n_lmdas: 正则化路径长度
        lmda_min_ratio: 最小正则化比例
        negative_penalty: 仅用于 fit_uni 的负系数惩罚
        verbose: 是否打印详细信息
        
    Returns:
        包含训练误差、测试误差、lambda路径等的字典
    """
    # 训练模型
    if verbose:
        print(f"Training {model_type} model...")
    
    if model_type == "unilasso":
        result = fit_unilasso(
            X=X_train,
            y=y_train,
            family=family,
            n_lmdas=n_lmdas,
            lmda_min_ratio=lmda_min_ratio,
            verbose=verbose
        )
    elif model_type == "uni":
        result = fit_uni(
            X=X_train,
            y=y_train,
            family=family,
            n_lmdas=n_lmdas,
            lmda_min_ratio=lmda_min_ratio,
            negative_penalty=negative_penalty,
            verbose=verbose
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # 获取正则化路径
    lmdas = result.lmdas
    
    # 在整个正则化路径上进行预测
    train_preds = predict_with_path(result, X_train)  # 形状: (n_train, n_lambdas)
    test_preds = predict_with_path(result, X_test)    # 形状: (n_test, n_lambdas)
    
    # 计算每个 lambda 对应的 MSE
    train_mses = np.array([compute_mse(y_train, train_preds[:, i]) for i in range(len(lmdas))])
    test_mses = np.array([compute_mse(y_test, test_preds[:, i]) for i in range(len(lmdas))])
    
    return {
        "model_type": model_type,
        "lmdas": lmdas,
        "neg_log_lmdas": -np.log(lmdas),
        "train_mses": train_mses,
        "test_mses": test_mses,
        "result": result
    }


def plot_train_test_error(
    exp_result: Dict,
    ax=None,
    title: Optional[str] = None,
    show_legend: bool = True
) -> plt.Axes:
    """
    绘制训练集和测试集误差曲线。
    
    Args:
        exp_result: experiment_train_test_error 的返回结果
        ax: matplotlib axes 对象，如果为 None 则创建新的 figure
        title: 图表标题
        show_legend: 是否显示图例
        
    Returns:
        matplotlib axes 对象
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    neg_log_lmdas = exp_result["neg_log_lmdas"]
    train_mses = exp_result["train_mses"]
    test_mses = exp_result["test_mses"]
    model_type = exp_result["model_type"]
    
    # 绘制训练集和测试集误差曲线
    ax.plot(neg_log_lmdas, train_mses, 'b-', linewidth=2, label='Train MSE', marker='o', markersize=4)
    ax.plot(neg_log_lmdas, test_mses, 'r-', linewidth=2, label='Test MSE', marker='s', markersize=4)
    
    # 设置标签和标题
    ax.set_xlabel(r"$-\log(\lambda)$", fontsize=12)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=12)
    
    if title is None:
        title = f"{model_type.upper()}: Train vs Test Error"
    ax.set_title(title, fontsize=14)
    
    # 添加网格和图例
    ax.grid(True, linestyle=':', alpha=0.6)
    if show_legend:
        ax.legend(loc='best', frameon=True)
    
    return ax


def run_comparison_experiment(
    n_samples: int = 1000,
    n_features: int = 20,
    sparsity: int = 5,
    signal_strength: float = 2.0,
    test_size: float = 0.3,
    random_state: int = 42,
    n_lmdas: int = 100,
    family: str = "gaussian",
    negative_penalty: float = 10.0,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    运行对比实验：比较 fit_uni 和 fit_unilasso 的训练/测试误差曲线。
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        sparsity: 非零系数的数量 (k-稀疏)
        signal_strength: 非零系数的大小
        test_size: 测试集比例（默认0.3，即三七分）
        random_state: 随机种子
        n_lmdas: 正则化路径长度
        family: 分布家族
        negative_penalty: fit_uni 的负系数惩罚
        save_path: 图片保存路径
        verbose: 是否打印详细信息
        
    Returns:
        (unilasso结果, uni结果) 的元组
    """
    if verbose:
        print("=" * 60)
        print("实验：对比 fit_uni 和 fit_unilasso 的训练/测试误差")
        print("=" * 60)
        print(f"数据设置: n_samples={n_samples}, n_features={n_features}")
        print(f"稀疏度: {sparsity} 个非零系数 (k-sparse)")
        print(f"信号强度: {signal_strength}")
        print(f"数据分割: 训练集={1-test_size:.0%}, 测试集={test_size:.0%}")
        print(f"正则化路径长度: {n_lmdas}")
        print("=" * 60)
    
    # 1. 生成 k-稀疏模拟数据
    if verbose:
        print("\n[1/4] 生成 k-稀疏模拟数据...")
    X, y, beta_true = simulate_sparse_gaussian_data(
        n=n_samples, 
        p=n_features, 
        sparsity=sparsity,
        signal_strength=signal_strength,
        seed=random_state
    )
    
    # 记录哪些特征真正有非零系数
    true_nonzero_indices = np.where(beta_true != 0)[0]
    if verbose:
        print(f"  真实非零系数位置: {true_nonzero_indices}")
        print(f"  真实非零系数值: {beta_true[true_nonzero_indices]}")
    
    # 2. 分割训练集和测试集（三七分）
    if verbose:
        print(f"\n[2/4] 分割数据集 (训练集: {1-test_size:.0%}, 测试集: {test_size:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"  训练集大小: {X_train.shape[0]}")
        print(f"  测试集大小: {X_test.shape[0]}")
    
    # 3. 训练 unilasso 模型并计算误差
    if verbose:
        print("\n[3/4] 训练 fit_unilasso 模型...")
    unilasso_result = experiment_train_test_error(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="unilasso",
        family=family,
        n_lmdas=n_lmdas,
        verbose=verbose
    )
    
    # 4. 训练 uni 模型并计算误差
    if verbose:
        print("\n[4/4] 训练 fit_uni 模型...")
    uni_result = experiment_train_test_error(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type="uni",
        family=family,
        n_lmdas=n_lmdas,
        negative_penalty=negative_penalty,
        verbose=verbose
    )
    
    # 5. 绘制对比图
    if verbose:
        print("\n绘制对比图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：unilasso
    plot_train_test_error(unilasso_result, ax=axes[0], title="fit_unilasso: Train vs Test Error")
    
    # 右图：uni
    plot_train_test_error(uni_result, ax=axes[1], title=f"fit_uni (penalty={negative_penalty}): Train vs Test Error")
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"图片已保存至: {save_path}")
    
    plt.show()
    
    # 6. 输出统计信息
    if verbose:
        print("\n" + "=" * 60)
        print("实验结果摘要")
        print("=" * 60)
        print(f"真实稀疏度: {sparsity} / {n_features} ({sparsity/n_features:.1%})")
        
        # 找到测试误差最小的点
        unilasso_best_idx = np.argmin(unilasso_result["test_mses"])
        uni_best_idx = np.argmin(uni_result["test_mses"])
        
        # 获取最佳模型的系数
        unilasso_best_coef = unilasso_result["result"].coefs[unilasso_best_idx] if unilasso_result["result"].coefs.ndim > 1 else unilasso_result["result"].coefs
        uni_best_coef = uni_result["result"].coefs[uni_best_idx] if uni_result["result"].coefs.ndim > 1 else uni_result["result"].coefs
        
        # 统计选中的特征数
        unilasso_n_selected = np.sum(np.abs(unilasso_best_coef) > 1e-6)
        uni_n_selected = np.sum(np.abs(uni_best_coef) > 1e-6)
        
        print(f"\nfit_unilasso:")
        print(f"  最佳 lambda: {unilasso_result['lmdas'][unilasso_best_idx]:.6f}")
        print(f"  最佳 -log(lambda): {unilasso_result['neg_log_lmdas'][unilasso_best_idx]:.4f}")
        print(f"  最佳测试 MSE: {unilasso_result['test_mses'][unilasso_best_idx]:.6f}")
        print(f"  选中特征数: {unilasso_n_selected}")
        print(f"  真实特征召回率: {np.sum(np.abs(unilasso_best_coef[true_nonzero_indices]) > 1e-6)}/{sparsity}")
        
        print(f"\nfit_uni (negative_penalty={negative_penalty}):")
        print(f"  最佳 lambda: {uni_result['lmdas'][uni_best_idx]:.6f}")
        print(f"  最佳 -log(lambda): {uni_result['neg_log_lmdas'][uni_best_idx]:.4f}")
        print(f"  最佳测试 MSE: {uni_result['test_mses'][uni_best_idx]:.6f}")
        print(f"  选中特征数: {uni_n_selected}")
        print(f"  真实特征召回率: {np.sum(np.abs(uni_best_coef[true_nonzero_indices]) > 1e-6)}/{sparsity}")
    
    return unilasso_result, uni_result


# 便捷函数：只运行一个模型的实验
def run_single_experiment(
    model_type: str = "unilasso",
    n_samples: int = 1000,
    n_features: int = 20,
    sparsity: int = 5,
    signal_strength: float = 2.0,
    test_size: float = 0.3,
    random_state: int = 42,
    n_lmdas: int = 100,
    negative_penalty: float = 10.0,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    运行单个模型的实验。
    
    Args:
        model_type: "unilasso" 或 "uni"
        n_samples: 样本数量
        n_features: 特征数量
        sparsity: 非零系数的数量 (k-稀疏)
        signal_strength: 非零系数的大小
        test_size: 测试集比例
        random_state: 随机种子
        n_lmdas: 正则化路径长度
        negative_penalty: 仅用于 fit_uni
        save_path: 图片保存路径
        verbose: 是否打印详细信息
        
    Returns:
        实验结果字典
    """
    if verbose:
        print(f"运行 {model_type} 实验...")
        print(f"数据稀疏度: {sparsity}/{n_features}")
    
    # 生成 k-稀疏数据
    X, y, beta_true = simulate_sparse_gaussian_data(
        n=n_samples, 
        p=n_features, 
        sparsity=sparsity,
        signal_strength=signal_strength,
        seed=random_state
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 运行实验
    result = experiment_train_test_error(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_type=model_type,
        n_lmdas=n_lmdas,
        negative_penalty=negative_penalty,
        verbose=verbose
    )
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_train_test_error(result, ax=ax)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"图片已保存至: {save_path}")
    
    plt.show()
    
    return result


if __name__ == "__main__":
    # 示例：运行对比实验
    print("示例：运行 fit_uni 和 fit_unilasso 的对比实验")
    print("数据模型: k-稀疏高斯线性回归")
    print()
    
    # 运行对比实验（k-稀疏设置）
    unilasso_res, uni_res = run_comparison_experiment(
        n_samples=500,
        n_features=20,      # 20个特征
        sparsity=5,         # 只有5个真正有非零系数
        signal_strength=2.0, # 非零系数大小
        test_size=0.3,      # 三七分
        random_state=42,
        n_lmdas=50,
        negative_penalty=10.0,
        save_path="experiment1_comparison.png",
        verbose=True
    )
