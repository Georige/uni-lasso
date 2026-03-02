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




# ==============================================================================
# 实验2：fit_uni 的 negative_penalty 调参实验
# ==============================================================================

def experiment2_negative_penalty_tuning(
    n_samples: int = 300,
    n_features: int = 15,
    sparsity: int = 4,
    signal_strength: float = 2.0,
    test_size: float = 0.3,
    random_state: int = 42,
    penalty_range: Tuple[float, float] = (0, 100),
    n_points: int = 11,
    n_lmdas: int = 40,
    save_path: Optional[str] = "experiment2_penalty_tuning.png",
    verbose: bool = True
) -> Dict:
    """
    实验2：调参 negative_penalty，分析其对模型效果的影响。
    
    关键发现：negative_penalty 只在小的 lambda 时产生明显差异（压制负系数）。
    因此本实验同时展示：
    1. 最佳测试 MSE vs penalty
    2. 负系数数量 vs penalty（在最小 lambda 处，即最不稀疏的模型）
    3. 选中特征数 vs penalty
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        sparsity: 非零系数数量
        signal_strength: 非零系数大小
        test_size: 测试集比例
        random_state: 随机种子
        penalty_range: negative_penalty 范围
        n_points: 采样点数
        n_lmdas: 每个模型的正则化路径长度
        save_path: 图片保存路径
        verbose: 是否打印详细信息
        
    Returns:
        实验结果字典
    """
    from tqdm import tqdm
    
    penalties = np.linspace(penalty_range[0], penalty_range[1], n_points)
    
    if verbose:
        print("=" * 70)
        print("实验2：fit_uni 的 negative_penalty 调参实验")
        print("=" * 70)
    
    # 生成数据（确保有负系数）
    np.random.seed(random_state)
    X, y, beta_true = simulate_sparse_gaussian_data(
        n=n_samples, p=n_features, sparsity=sparsity, 
        signal_strength=signal_strength, seed=random_state
    )
    
    # 将一半的非零系数设为负，确保有负系数被测试
    nonzero_idx = np.where(beta_true != 0)[0]
    for i in range(len(nonzero_idx) // 2):
        beta_true[nonzero_idx[i]] = -abs(beta_true[nonzero_idx[i]])
    y = X @ beta_true + np.random.normal(size=n_samples)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    true_nonzero = np.where(beta_true != 0)[0]
    true_negative = np.where(beta_true < 0)[0]
    
    if verbose:
        print(f"\n数据设置:")
        print(f"  样本数: {n_samples}, 特征数: {n_features}")
        print(f"  真实非零系数: {len(true_nonzero)} 个")
        print(f"  真实负系数: {len(true_negative)} 个，位置: {true_negative}")
        print(f"  惩罚系数范围: [{penalty_range[0]:.1f}, {penalty_range[1]:.1f}]，共 {n_points} 个点")
    
    # 存储结果
    results = {
        "penalties": [],
        "best_test_mses": [],
        "best_train_mses": [],
        "best_lambdas": [],
        "n_selected": [],
        "n_negative_at_best": [],
        "n_negative_at_small_lambda": [],
    }
    
    iterator = tqdm(penalties, desc="Testing penalties") if verbose else penalties
    
    for penalty in iterator:
        try:
            result = fit_uni(
                X=X_train, y=y_train,
                n_lmdas=n_lmdas,
                negative_penalty=float(penalty),
                verbose=False
            )
            
            # 在整个路径上预测并计算MSE
            test_preds = predict_with_path(result, X_test)
            train_preds = predict_with_path(result, X_train)
            
            test_mses = np.array([compute_mse(y_test, test_preds[:, j]) for j in range(len(result.lmdas))])
            train_mses = np.array([compute_mse(y_train, train_preds[:, j]) for j in range(len(result.lmdas))])
            
            best_idx = np.argmin(test_mses)
            
            # 获取不同位置的系数
            best_coef = result.coefs[best_idx] if result.coefs.ndim > 1 else result.coefs
            small_lambda_coef = result.coefs[-1] if result.coefs.ndim > 1 else result.coefs
            
            results["penalties"].append(penalty)
            results["best_test_mses"].append(test_mses[best_idx])
            results["best_train_mses"].append(train_mses[best_idx])
            results["best_lambdas"].append(result.lmdas[best_idx])
            results["n_selected"].append(np.sum(np.abs(best_coef) > 1e-6))
            results["n_negative_at_best"].append(np.sum(best_coef < -1e-6))
            results["n_negative_at_small_lambda"].append(np.sum(small_lambda_coef < -1e-6))
            
        except Exception as e:
            if verbose:
                print(f"\n警告: penalty={penalty} 失败: {e}")
            continue
    
    for key in results:
        results[key] = np.array(results[key])
    
    # 绘制结果（2x2子图）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 最佳测试MSE
    ax = axes[0, 0]
    ax.plot(results["penalties"], results["best_test_mses"], 'bo-', markersize=8, linewidth=2)
    ax.set_xlabel("Negative Penalty", fontsize=12)
    ax.set_ylabel("Best Test MSE", fontsize=12)
    ax.set_title("Best Test MSE vs Penalty", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    best_idx = np.argmin(results["best_test_mses"])
    ax.plot(results["penalties"][best_idx], results["best_test_mses"][best_idx], 
            'r*', markersize=15, label=f'Best: penalty={results["penalties"][best_idx]:.1f}')
    ax.legend()
    
    # 图2: 选中特征数
    ax = axes[0, 1]
    ax.plot(results["penalties"], results["n_selected"], 'gs-', markersize=8, linewidth=2)
    ax.axhline(y=len(true_nonzero), color='r', linestyle='--', alpha=0.5, 
               label=f'True non-zero: {len(true_nonzero)}')
    ax.set_xlabel("Negative Penalty", fontsize=12)
    ax.set_ylabel("Selected Features", fontsize=12)
    ax.set_title("Selected Features vs Penalty", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 图3: 负系数数量（最佳lambda处）
    ax = axes[1, 0]
    ax.plot(results["penalties"], results["n_negative_at_best"], 'r^-', markersize=8, linewidth=2)
    ax.axhline(y=len(true_negative), color='k', linestyle='--', alpha=0.5, 
               label=f'True negative: {len(true_negative)}')
    ax.set_xlabel("Negative Penalty", fontsize=12)
    ax.set_ylabel("Negative Coefficients", fontsize=12)
    ax.set_title("Negative Coefficients at Best Lambda", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    # 图4: 负系数数量（最小lambda处 - 最关键！）
    ax = axes[1, 1]
    ax.plot(results["penalties"], results["n_negative_at_small_lambda"], 'm^-', markersize=8, linewidth=2)
    ax.axhline(y=len(true_negative), color='k', linestyle='--', alpha=0.5, 
               label=f'True negative: {len(true_negative)}')
    ax.set_xlabel("Negative Penalty", fontsize=12)
    ax.set_ylabel("Negative Coefficients", fontsize=12)
    ax.set_title("Negative Coefficients at Smallest Lambda", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n图片已保存至: {save_path}")
    
    plt.show()
    
    # 输出统计
    if verbose:
        print("\n" + "=" * 70)
        print("实验2结果摘要")
        print("=" * 70)
        print(f"{'Penalty':<10} {'Test MSE':<12} {'Selected':<10} {'Neg@Best':<12} {'Neg@Small'}")
        print("-" * 70)
        for i in range(len(results["penalties"])):
            print(f"{results['penalties'][i]:<10.1f} {results['best_test_mses'][i]:<12.6f} "
                  f"{results['n_selected'][i]:<10} {results['n_negative_at_best'][i]:<12} "
                  f"{results['n_negative_at_small_lambda'][i]}")
        
        print(f"\n关键发现:")
        print(f"  - Penalty=0 时，最小lambda处有 {results['n_negative_at_small_lambda'][0]} 个负系数")
        print(f"  - Penalty={results['penalties'][-1]:.0f} 时，最小lambda处有 {results['n_negative_at_small_lambda'][-1]} 个负系数")
        print(f"  - 真实负系数: {len(true_negative)} 个")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        # 运行实验2
        print("运行实验2：fit_uni 的 negative_penalty 调参")
        print()
        
        results = experiment2_negative_penalty_tuning(
            n_samples=300,
            n_features=15,
            sparsity=3,
            signal_strength=2.0,
            penalty_range=(0, 100),
            n_points=6,
            n_lmdas=40,
            save_path="experiment2_penalty_tuning.png",
            verbose=True
        )
    else:
        # 默认运行实验1
        print("示例：运行 fit_uni 和 fit_unilasso 的对比实验")
        print("数据模型: k-稀疏高斯线性回归")
        print()
        
        unilasso_res, uni_res = run_comparison_experiment(
            n_samples=500,
            n_features=10,
            sparsity=3,
            signal_strength=2.0,
            test_size=0.3,
            random_state=42,
            n_lmdas=40,
            negative_penalty=10.0,
            save_path="experiment1_comparison.png",
            verbose=True
        )


# ==============================================================================
# 实验3：negative_penalty 对支持集（active sets）的影响分析
# ==============================================================================

def experiment3_active_set_analysis(
    n_samples: int = 400,
    n_features: int = 20,
    sparsity: int = 5,
    signal_strength: float = 2.0,
    test_size: float = 0.3,
    random_state: int = 42,
    penalty_range: Tuple[float, float] = (0, 100),
    n_points: int = 21,
    fixed_lambda: Optional[float] = None,
    save_path: Optional[str] = "experiment3_active_set_analysis.png",
    verbose: bool = True
) -> Dict:
    """
    实验3：分析 negative_penalty 对模型支持集（active sets）的影响。
    
    在固定lambda下，比较不同negative_penalty对应的选中特征数，
    并与标准Lasso和fit_unilasso进行对比。
    
    Args:
        n_samples: 样本数量
        n_features: 特征数量
        sparsity: 非零系数数量
        signal_strength: 非零系数大小
        test_size: 测试集比例
        random_state: 随机种子
        penalty_range: negative_penalty 范围
        n_points: 采样点数
        fixed_lambda: 固定的lambda值（None则使用中等大小的lambda）
        save_path: 图片保存路径
        verbose: 是否打印详细信息
        
    Returns:
        包含分析结果的字典
    """
    import adelie as ad
    from tqdm import tqdm
    
    penalties = np.linspace(penalty_range[0], penalty_range[1], n_points)
    
    if verbose:
        print("=" * 70)
        print("实验3：negative_penalty 对支持集的影响分析")
        print("=" * 70)
    
    # 生成数据
    np.random.seed(random_state)
    X, y, beta_true = simulate_sparse_gaussian_data(
        n=n_samples, p=n_features, sparsity=sparsity,
        signal_strength=signal_strength, seed=random_state
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    true_nonzero = np.where(beta_true != 0)[0]
    
    # 准备fit_uni所需的数据（获取loo_fits和lambda路径）
    from unilasso.uni_lasso import _prepare_unilasso_input, _configure_lmda_path, fit_unilasso
    X_proc, y_proc, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, _, _, zero_var_idx = \
        _prepare_unilasso_input(X_train, y_train, "gaussian", None)
    
    lambda_path = _configure_lmda_path(X=loo_fits, y=y_train, family="gaussian", n_lmdas=50, lmda_min_ratio=1e-4)
    
    # 确定固定的lambda值
    if fixed_lambda is None:
        # 使用中等大小的lambda（第30%位置的lambda）
        fixed_lambda = lambda_path[int(len(lambda_path) * 0.3)]
    
    # 找到最接近fixed_lambda的索引
    closest_idx = np.argmin(np.abs(lambda_path - fixed_lambda))
    actual_lambda = lambda_path[closest_idx]
    
    if verbose:
        print(f"\n数据设置:")
        print(f"  样本数: {n_samples}, 特征数: {n_features}")
        print(f"  真实非零系数: {len(true_nonzero)} 个")
        print(f"\n固定的 lambda: {actual_lambda:.6f} (目标: {fixed_lambda:.6f})")
        print(f"惩罚系数范围: [{penalty_range[0]:.1f}, {penalty_range[1]:.1f}]，共 {n_points} 个点")
    
    # 1. 训练标准Lasso模型（使用adelie）
    if verbose:
        print("\n[1/3] 训练标准Lasso模型...")
    glm_y = ad.glm.gaussian(y_train)
    lasso_model = ad.grpnet(
        X=np.asfortranarray(X_train),
        glm=glm_y,
        intercept=True,
        lmda_path_size=100,
        min_ratio=1e-4
    )
    # 找到最接近的lambda
    lasso_closest_idx = np.argmin(np.abs(np.array(lasso_model.lmdas) - actual_lambda))
    lasso_coef = lasso_model.betas.toarray()[lasso_closest_idx]
    lasso_active_set = np.sum(np.abs(lasso_coef) > 1e-6)
    lasso_lambda = lasso_model.lmdas[lasso_closest_idx]
    
    if verbose:
        print(f"  Lasso选中特征数: {lasso_active_set} (lambda={lasso_lambda:.6f})")
    
    # 2. 训练fit_unilasso模型
    if verbose:
        print("\n[2/3] 训练fit_unilasso模型...")
    unilasso_result = fit_unilasso(
        X=X_train, y=y_train,
        n_lmdas=50,
        verbose=False
    )
    unilasso_closest_idx = np.argmin(np.abs(unilasso_result.lmdas - actual_lambda))
    unilasso_coef = unilasso_result.coefs[unilasso_closest_idx] if unilasso_result.coefs.ndim > 1 else unilasso_result.coefs
    unilasso_active_set = np.sum(np.abs(unilasso_coef) > 1e-6)
    unilasso_lambda = unilasso_result.lmdas[unilasso_closest_idx]
    
    if verbose:
        print(f"  UniLasso选中特征数: {unilasso_active_set} (lambda={unilasso_lambda:.6f})")
    
    # 3. 测试不同negative_penalty下的支持集
    if verbose:
        print("\n[3/3] 测试不同negative_penalty下的支持集...")
    
    results = {
        "penalties": [],
        "active_sets": [],
        "actual_lambda": actual_lambda,
        "lasso_active_set": lasso_active_set,
        "unilasso_active_set": unilasso_active_set,
    }
    
    iterator = tqdm(penalties, desc="Testing penalties") if verbose else penalties
    
    for penalty in iterator:
        try:
            result = fit_uni(
                X=X_train, y=y_train,
                n_lmdas=50,
                negative_penalty=float(penalty),
                verbose=False
            )
            
            # 找到最接近fixed_lambda的系数
            closest_idx = np.argmin(np.abs(result.lmdas - actual_lambda))
            coef = result.coefs[closest_idx] if result.coefs.ndim > 1 else result.coefs
            active_set = np.sum(np.abs(coef) > 1e-6)
            
            results["penalties"].append(penalty)
            results["active_sets"].append(active_set)
            
        except Exception as e:
            if verbose:
                print(f"\n警告: penalty={penalty} 失败: {e}")
            continue
    
    for key in ["penalties", "active_sets"]:
        results[key] = np.array(results[key])
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制fit_uni曲线
    ax.plot(results["penalties"], results["active_sets"], 'bo-', markersize=8, 
            linewidth=2, label='fit_uni (different penalties)', alpha=0.7)
    
    # 绘制标准Lasso参考线（红色水平线）
    ax.axhline(y=lasso_active_set, color='red', linestyle='--', linewidth=2,
               label=f'Standard Lasso: {lasso_active_set} features')
    ax.scatter([penalty_range[1] * 0.95], [lasso_active_set], color='red', s=200, 
               marker='s', zorder=5, edgecolors='black', linewidths=2)
    
    # 绘制UniLasso参考线（绿色水平线）
    ax.axhline(y=unilasso_active_set, color='green', linestyle='--', linewidth=2,
               label=f'UniLasso: {unilasso_active_set} features')
    ax.scatter([penalty_range[1] * 0.95], [unilasso_active_set], color='green', s=200, 
               marker='^', zorder=5, edgecolors='black', linewidths=2)
    
    # 绘制真实稀疏度参考线
    ax.axhline(y=len(true_nonzero), color='gray', linestyle=':', linewidth=2, alpha=0.5,
               label=f'True sparsity: {len(true_nonzero)}')
    
    ax.set_xlabel("Negative Penalty", fontsize=14)
    ax.set_ylabel("Active Set Size (Number of Selected Features)", fontsize=14)
    ax.set_title(f"Active Set vs Negative Penalty (Fixed λ={actual_lambda:.4f})", fontsize=15)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best', fontsize=11, frameon=True)
    
    # 设置y轴范围，留出一些空间
    y_min = min(min(results["active_sets"]), lasso_active_set, unilasso_active_set, len(true_nonzero)) - 1
    y_max = max(max(results["active_sets"]), lasso_active_set, unilasso_active_set, len(true_nonzero)) + 2
    ax.set_ylim(max(0, y_min), y_max)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"\n图片已保存至: {save_path}")
    
    plt.show()
    
    # 输出统计
    if verbose:
        print("\n" + "=" * 70)
        print("实验3结果摘要")
        print("=" * 70)
        print(f"固定lambda: {actual_lambda:.6f}")
        print(f"真实非零系数: {len(true_nonzero)}")
        print(f"\n参考模型:")
        print(f"  标准Lasso选中: {lasso_active_set} 个特征")
        print(f"  UniLasso选中: {unilasso_active_set} 个特征")
        print(f"\nfit_uni随penalty变化:")
        print(f"  Penalty=0时: {results['active_sets'][0]} 个特征")
        print(f"  Penalty={results['penalties'][-1]:.0f}时: {results['active_sets'][-1]} 个特征")
        
        # 找到最接近真实稀疏度的penalty
        closest_to_truth = np.argmin(np.abs(results["active_sets"] - len(true_nonzero)))
        print(f"  最接近真实稀疏度的penalty: {results['penalties'][closest_to_truth]:.1f} "
              f"({results['active_sets'][closest_to_truth]} 个特征)")
    
    return results


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "2":
        # 运行实验2
        print("运行实验2：fit_uni 的 negative_penalty 调参")
        print()
        
        results = experiment2_negative_penalty_tuning(
            n_samples=300,
            n_features=15,
            sparsity=4,
            signal_strength=2.0,
            penalty_range=(0, 100),
            n_points=6,
            n_lmdas=40,
            save_path="experiment2_penalty_tuning.png",
            verbose=True
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "3":
        # 运行实验3
        print("运行实验3：negative_penalty 对支持集的影响分析")
        print()
        
        results = experiment3_active_set_analysis(
            n_samples=200,
            n_features=10,
            sparsity=3,
            signal_strength=2.0,
            penalty_range=(0, 100),
            n_points=6,
            fixed_lambda=None,  # 自动选择中等lambda
            save_path="experiment3_active_set_analysis.png",
            verbose=True
        )
    else:
        # 默认运行实验1
        print("示例：运行 fit_uni 和 fit_unilasso 的对比实验")
        print("数据模型: k-稀疏高斯线性回归")
        print()
        
        unilasso_res, uni_res = run_comparison_experiment(
            n_samples=500,
            n_features=10,
            sparsity=3,
            signal_strength=2.0,
            test_size=0.3,
            random_state=42,
            n_lmdas=40,
            negative_penalty=10.0,
            save_path="experiment1_comparison.png",
            verbose=True
        )
