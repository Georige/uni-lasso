"""
================================================================================
探究：negative_penalty 在 SoftUniLasso 中的作用
================================================================================

本演示展示 negative_penalty 参数如何影响拟合系数 theta 的行为。
"""

import numpy as np
import matplotlib.pyplot as plt
from unilasso.uni_lasso import _fit_pytorch_lasso_path


def create_test_data(n=100, p=5, seed=42):
    """创建测试数据，确保某些特征与目标变量负相关"""
    np.random.seed(seed)
    loo_fits = np.random.randn(n, p)
    # 构造 y，使其与某些列正相关，与某些列负相关
    true_theta = np.array([2.0, -1.5, 0.8, -2.0, 0.5])
    y = loo_fits @ true_theta + 0.3 * np.random.randn(n)
    return loo_fits, y, true_theta


def main():
    # 创建数据
    loo_fits, y, true_theta = create_test_data(n=100, p=5, seed=42)
    
    print("=" * 80)
    print("探究：negative_penalty 对 theta 的影响")
    print("=" * 80)
    print(f"\n数据: n={loo_fits.shape[0]}, p={loo_fits.shape[1]}")
    print(f"真实 theta: {true_theta}")
    
    # OLS 解
    theta_ols = np.linalg.lstsq(loo_fits, y, rcond=None)[0]
    print(f"OLS theta:  {np.round(theta_ols, 4)}")
    print(f"负系数数量: {np.sum(theta_ols < 0)}")
    
    # 测试不同 negative_penalty
    lmdas = np.array([0.01])
    penalties = [0, 1, 5, 10, 50, 100, 1000]
    results = {}
    
    print(f"\n不同 negative_penalty 下的 theta（lambda={lmdas[0]}）：")
    print("-" * 80)
    print(f"{'Penalty':>10} | {'Theta':>50} | {'负系数':>8}")
    print("-" * 80)
    
    for penalty in penalties:
        thetas, _ = _fit_pytorch_lasso_path(
            loo_fits, y, lmdas, 
            negative_penalty=penalty,
            lr=0.01, max_epochs=5000, tol=1e-6
        )
        theta = thetas[0]
        results[penalty] = theta
        neg_count = np.sum(theta < -1e-6)
        print(f"{penalty:>10} | {str(np.round(theta, 3)):>50} | {neg_count:>8}")
    
    print("-" * 80)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：各 theta 随 penalty 变化
    ax1 = axes[0]
    theta_matrix = np.array([results[p] for p in penalties])
    for j in range(theta_matrix.shape[1]):
        ax1.plot(penalties, theta_matrix[:, j], marker='o', label=f'θ{j+1}')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xlabel('negative_penalty')
    ax1.set_ylabel('theta')
    ax1.set_title('Effect of negative_penalty on theta')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 右图：负系数数量
    ax2 = axes[1]
    neg_counts = [np.sum(theta_matrix[i] < -1e-6) for i in range(len(penalties))]
    ax2.plot(penalties, neg_counts, marker='s', color='crimson', linewidth=2)
    ax2.set_xlabel('negative_penalty')
    ax2.set_ylabel('Number of negative coefficients')
    ax2.set_title('Negative coefficients eliminated by penalty')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('negative_penalty_demo.png', dpi=150)
    print("\n可视化已保存: negative_penalty_demo.png")
    plt.show()


if __name__ == "__main__":
    main()
