"""
高级数据生成模块

包含四种复杂数据生成场景：
1. AR(1)相关性数据：系数全正，变量间有自回归相关性
2. 高维稀疏数据：n=300, p=1000，相关性0.5，高中低信噪比
3. 符号不一致数据：X_2 = X_1 + 噪声，用于测试符号反转
4. 因子模型数据：潜变量生成X，X生成Y
"""

import numpy as np
from scipy.linalg import toeplitz
from typing import Tuple, Optional, Literal


def generate_ar1_data(
    n: int = 300,
    p: int = 50,
    rho: float = 0.7,
    sparsity: int = 10,
    beta_min: float = 0.5,
    beta_max: float = 2.0,
    noise_std: float = 1.0,
    seed: Optional[int] = 123
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成AR(1)相关性数据。
    
    特征间有自回归相关性：Cov(X_i, X_j) = rho^|i-j|
    所有非零系数都是正数。
    
    Args:
        n: 样本数量
        p: 特征数量
        rho: AR(1)相关系数 (0 < rho < 1)
        sparsity: 非零系数数量
        beta_min, beta_max: 系数的范围（都是正数）
        noise_std: 噪声标准差
        seed: 随机种子
        
    Returns:
        X: 特征矩阵 (n, p)
        y: 响应变量 (n,)
        beta_true: 真实系数 (p,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成AR(1)协方差矩阵
    cov_matrix = np.power(rho, np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]))
    
    # 生成特征 X ~ N(0, cov_matrix)
    X = np.random.multivariate_normal(np.zeros(p), cov_matrix, size=n)
    
    # 生成稀疏正系数
    beta_true = np.zeros(p)
    nonzero_indices = np.random.choice(p, size=sparsity, replace=False)
    # 所有系数都是正数
    beta_true[nonzero_indices] = np.random.uniform(beta_min, beta_max, size=sparsity)
    
    # 生成响应变量
    y = X @ beta_true + np.random.normal(0, noise_std, size=n)
    
    return X, y, beta_true


def generate_highdim_correlated_data(
    n: int = 300,
    p: int = 1000,
    sparsity: int = 20,
    correlation: float = 0.5,
    snr_level: Literal['low', 'medium', 'high'] = 'medium',
    signal_strength: float = 2.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    生成高维稀疏相关数据（信息不足情况）。
    
    n=300, p=1000，变量间有固定相关性，可选择信噪比。
    
    Args:
        n: 样本数量（默认300）
        p: 特征数量（默认1000）
        sparsity: 非零系数数量
        correlation: 变量间相关系数（默认0.5）
        snr_level: 信噪比级别 ('low', 'medium', 'high')
        signal_strength: 信号强度
        seed: 随机种子
        
    Returns:
        X: 特征矩阵 (n, p)
        y: 响应变量 (n,)
        beta_true: 真实系数 (p,)
        snr: 实际信噪比
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成具有指定相关性的特征
    # 使用块对角结构，每块内部有相关性
    block_size = 10
    n_blocks = p // block_size
    
    cov_blocks = []
    for _ in range(n_blocks):
        block = np.ones((block_size, block_size)) * correlation
        np.fill_diagonal(block, 1.0)
        cov_blocks.append(block)
    
    # 构建块对角协方差矩阵
    cov_matrix = np.block([[cov_blocks[i] if i == j else np.zeros((block_size, block_size)) 
                            for j in range(n_blocks)] for i in range(n_blocks)])
    
    # 如果p不是block_size的整数倍，补充剩余部分
    if p % block_size != 0:
        remaining = p % block_size
        remaining_cov = np.eye(remaining)
        cov_matrix = np.block([[cov_matrix, np.zeros((cov_matrix.shape[0], remaining))],
                               [np.zeros((remaining, cov_matrix.shape[1])), remaining_cov]])
    
    # 生成特征
    X = np.random.multivariate_normal(np.zeros(p), cov_matrix, size=n)
    
    # 生成稀疏系数（混合正负）
    beta_true = np.zeros(p)
    nonzero_indices = np.random.choice(p, size=sparsity, replace=False)
    beta_values = np.random.uniform(-signal_strength, signal_strength, size=sparsity)
    # 确保有正有负
    beta_values[:sparsity//2] = -np.abs(beta_values[:sparsity//2])
    beta_values[sparsity//2:] = np.abs(beta_values[sparsity//2:])
    beta_true[nonzero_indices] = beta_values
    
    # 生成纯净信号
    signal = X @ beta_true
    signal_var = np.var(signal)
    
    # 根据信噪比确定噪声水平
    snr_map = {'low': 0.5, 'medium': 2.0, 'high': 10.0}
    target_snr = snr_map[snr_level]
    
    # SNR = signal_var / noise_var
    noise_var = signal_var / target_snr
    noise_std = np.sqrt(noise_var)
    
    y = signal + np.random.normal(0, noise_std, size=n)
    
    # 计算实际SNR
    actual_snr = signal_var / np.var(y - signal)
    
    return X, y, beta_true, actual_snr


def generate_sign_inconsistent_data(
    n: int = 300,
    p: int = 20,
    noise_X: float = 0.1,
    noise_y: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成符号不一致数据，用于测试模型是否能正确处理符号反转。
    
    X_2 = X_1 + 轻微噪声 (高度相关)
    真实模型: y = 1*X_1 - 0.5*X_2 + 0*X_3 + ... + noise
    
    由于X_1和X_2高度相关，单变量回归可能给出同号系数，
    但真实模型需要异号系数。
    
    Args:
        n: 样本数量
        p: 特征数量（至少2个）
        noise_X: X_2相对于X_1的噪声水平
        noise_y: y的噪声水平
        seed: 随机种子
        
    Returns:
        X: 特征矩阵 (n, p)
        y: 响应变量 (n,)
        beta_true: 真实系数 (p,)
    """
    if seed is not None:
        np.random.seed(seed)
    
    assert p >= 2, "p must be at least 2"
    
    # 生成基础特征 X_1
    X1 = np.random.normal(0, 1, size=n)
    
    # X_2 = X_1 + 噪声 (高度相关但不完全相同)
    X2 = X1 + np.random.normal(0, noise_X, size=n)
    
    # 其余特征独立
    X_rest = np.random.normal(0, 1, size=(n, p-2))
    
    X = np.column_stack([X1, X2, X_rest])
    
    # 真实系数: [1, -0.5, 0, 0, ...]
    beta_true = np.zeros(p)
    beta_true[0] = 1.0
    beta_true[1] = -0.5
    
    # 生成响应变量
    y = X @ beta_true + np.random.normal(0, noise_y, size=n)
    
    return X, y, beta_true


def generate_factor_model_data(
    n: int = 300,
    p: int = 50,
    n_factors: int = 5,
    sparsity: int = 10,
    factor_loading_range: Tuple[float, float] = (0.5, 1.5),
    signal_strength: float = 2.0,
    noise_std: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    生成因子模型数据。
    
    潜变量 F 生成观测变量 X，然后 X 生成 Y。
    模型:
        F ~ N(0, I)  (n_factors个潜变量)
        X = F @ B + epsilon_X  (B是因子载荷矩阵)
        y = X @ beta + epsilon_y
    
    Args:
        n: 样本数量
        p: 特征数量（观测变量数）
        n_factors: 潜变量数量
        sparsity: 非零系数数量
        factor_loading_range: 因子载荷的范围
        signal_strength: 信号强度
        noise_std: 噪声标准差
        seed: 随机种子
        
    Returns:
        X: 特征矩阵 (n, p)
        y: 响应变量 (n,)
        beta_true: 真实系数 (p,)
        F: 潜变量矩阵 (n, n_factors)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 生成潜变量 F
    F = np.random.normal(0, 1, size=(n, n_factors))
    
    # 生成因子载荷矩阵 B (n_factors, p)
    B = np.random.uniform(
        factor_loading_range[0], 
        factor_loading_range[1], 
        size=(n_factors, p)
    )
    # 随机赋予正负号
    B = B * np.random.choice([-1, 1], size=B.shape)
    
    # 生成观测变量 X = F @ B + noise
    X_signal = F @ B
    X_noise = np.random.normal(0, 0.5, size=(n, p))  # 较小的观测噪声
    X = X_signal + X_noise
    
    # 生成稀疏系数
    beta_true = np.zeros(p)
    nonzero_indices = np.random.choice(p, size=sparsity, replace=False)
    beta_values = np.random.uniform(-signal_strength, signal_strength, size=sparsity)
    beta_true[nonzero_indices] = beta_values
    
    # 生成响应变量
    y = X @ beta_true + np.random.normal(0, noise_std, size=n)
    
    return X, y, beta_true, F


def get_data_generator(data_type: str):
    """
    根据数据类型返回对应的数据生成函数。
    
    Args:
        data_type: 数据类型 ('ar1', 'highdim', 'sign_inconsistent', 'factor')
        
    Returns:
        对应的数据生成函数
    """
    generators = {
        'ar1': generate_ar1_data,
        'highdim': generate_highdim_correlated_data,
        'sign_inconsistent': generate_sign_inconsistent_data,
        'factor': generate_factor_model_data,
    }
    
    if data_type not in generators:
        raise ValueError(f"Unknown data type: {data_type}. Available: {list(generators.keys())}")
    
    return generators[data_type]


if __name__ == "__main__":
    # 测试数据生成器
    print("测试数据生成器...")
    print("=" * 70)
    
    # 测试 AR(1)
    print("\n1. AR(1) 数据:")
    X, y, beta = generate_ar1_data(n=100, p=20, seed=42)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Nonzero beta: {np.sum(beta != 0)}")
    print(f"   All positive: {np.all(beta[beta != 0] > 0)}")
    print(f"   X correlation (0,1): {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.3f}")
    
    # 测试高维
    print("\n2. 高维稀疏数据:")
    X, y, beta, snr = generate_highdim_correlated_data(
        n=200, p=500, snr_level='medium', seed=42
    )
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   Nonzero beta: {np.sum(beta != 0)}")
    print(f"   Actual SNR: {snr:.2f}")
    
    # 测试符号不一致
    print("\n3. 符号不一致数据:")
    X, y, beta = generate_sign_inconsistent_data(n=100, p=10, seed=42)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    print(f"   True beta: {beta[:3]}")
    print(f"   Correlation(X1, X2): {np.corrcoef(X[:, 0], X[:, 1])[0, 1]:.3f}")
    
    # 测试因子模型
    print("\n4. 因子模型数据:")
    X, y, beta, F = generate_factor_model_data(n=100, p=30, n_factors=3, seed=42)
    print(f"   X shape: {X.shape}, F shape: {F.shape}, y shape: {y.shape}")
    print(f"   Nonzero beta: {np.sum(beta != 0)}")
    print(f"   X correlation structure (first 3x3):")
    print(np.corrcoef(X[:, :3].T).round(2))
