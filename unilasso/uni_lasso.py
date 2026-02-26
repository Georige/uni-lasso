"""
UniLasso: Univariate-Guided Sparse Regression

This module implements core LOO functions for Univariate-Guided Lasso regression.

Reference: https://arxiv.org/abs/2501.18360
"""



import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt
import adelie as ad


from typing import List, Optional, Tuple, Union, Callable
import logging

from .univariate_regression import fit_loo_univariate_models
from .config import VALID_FAMILIES
from .utils import warn_zero_variance, warn_removed_lmdas



# ￥偷梁换柱
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

class CustomSoftPenaltyLasso:
    """一个伪装成 ad.grpnet 的自定义求解器"""
    
    def __init__(self, X, y, lmda_path, alpha_penalty=100.0):
        self.X = X
        self.y = y
        self.lmdas = lmda_path
        self.alpha = alpha_penalty # 控制对负数的软惩罚力度
        
        # 为了兼容下游代码，我们需要准备好 betas 和 intercepts 属性
        self.betas = None 
        self.intercepts = np.zeros(len(lmda_path))
        
        # 立即开始拟合
        self._fit()

    def _custom_loss(self, theta, lmda):
        """这就是你想要的全新 Loss Function！"""
        # 1. 基础误差 (MSE)
        residuals = self.y - self.X @ theta
        mse = np.mean(residuals ** 2) / 2
        
        # 2. L1 正则化 (Lasso)
        l1_penalty = lmda * np.sum(np.abs(theta))
        
        # 3. 你的专属创新：负数软惩罚 (Soft Penalty for negatives)
        # 找出所有小于 0 的 theta，计算其平方和作为惩罚
        neg_theta = np.minimum(0, theta) 
        soft_penalty = self.alpha * np.sum(neg_theta ** 2)
        
        # 返回总 Loss
        return mse + l1_penalty + soft_penalty

    def _fit(self):
        n_features = self.X.shape[1]
        all_betas = []
        
        # 初始猜测值为全 0
        current_theta = np.zeros(n_features)
        
        # 遍历所有的 lambda 进行求解
        for i, lmda in enumerate(self.lmdas):
            # 使用 L-BFGS-B 优化算法寻找最小 Loss
            res = minimize(
                fun=self._custom_loss, 
                x0=current_theta, 
                args=(lmda,), 
                method='L-BFGS-B'
            )
            current_theta = res.x
            all_betas.append(current_theta)
        
        # 将结果转换为稀疏矩阵，完美伪装成 C++ 引擎的输出格式！
        self.betas = csr_matrix(np.array(all_betas))
        



# Configure logger
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------


import numpy as np
from typing import Optional, Callable
import adelie as ad


# 父类结果储存
class UniLassoResultBase:
    """
    Base class for UniLasso results, encapsulating model outputs.
    """

    def __init__(self, 
                 coefs: np.ndarray, 
                 intercept: np.ndarray, 
                 family: str,
                 gamma: np.ndarray, 
                 gamma_intercept: np.ndarray, 
                 beta: np.ndarray, 
                 beta_intercepts: np.ndarray, 
                 lasso_model: ad.grpnet, 
                 lmdas: np.ndarray):
        """
        Initializes the base UniLasso result object.

        Parameters:
        - coefs (np.ndarray): Coefficients of the univariate-guided lasso.
        - intercept (np.ndarray): Intercept of the univariate-guided lasso.
        - family (str): Family of the response variable ('gaussian', 'binomial', 'cox').
        - gamma (np.ndarray): Hidden gamma coefficients.
        - gamma_intercept (np.ndarray): Hidden gamma intercept.
        - beta (np.ndarray): Hidden beta coefficients.
        - beta_intercepts (np.ndarray): Hidden beta intercepts.
        - lasso_model (ad.grpnet): The fitted Lasso model.
        - lmdas (np.ndarray): Regularization path.
        """
        self.coefs = coefs
        self.intercept = intercept
        self.family = family
        self._gamma = gamma  
        self._gamma_intercept = gamma_intercept  
        self._beta = beta  
        self._beta_intercepts = beta_intercepts  
        self.lasso_model = lasso_model
        self.lmdas = lmdas

    def get_gamma(self) -> np.ndarray:
        """Returns the hidden gamma coefficients."""
        return self._gamma

    def get_gamma_intercept(self) -> np.ndarray:
        """Returns the hidden gamma intercept."""
        return self._gamma_intercept

    def get_beta(self) -> np.ndarray:
        """Returns the hidden beta coefficients."""
        return self._beta

    def get_beta_intercepts(self) -> np.ndarray:
        """Returns the hidden beta intercepts."""
        return self._beta_intercepts

    def __repr__(self):
        """Custom string representation of the result object."""
        return (f"{self.__class__.__name__}(coefs={self.coefs.shape}, "
                f"intercept={self.intercept.shape}, "
                f"lasso_model={type(self.lasso_model).__name__}, "
                f"lmdas={self.lmdas.shape})")


class UniLassoResult(UniLassoResultBase):
    """
    Class for storing standard UniLasso results.
    """
    pass


class UniLassoCVResult(UniLassoResultBase):
    """
    Class for storing cross-validation UniLasso results.
    """

    def __init__(self, 
                 coefs: np.ndarray, 
                 intercept: np.ndarray, 
                 family: str,
                 gamma: np.ndarray, 
                 gamma_intercept: np.ndarray, 
                 beta: np.ndarray, 
                 beta_intercepts: np.ndarray, 
                 lasso_model: ad.grpnet, 
                 lmdas: np.ndarray,
                 avg_losses: np.ndarray, 
                 cv_plot: Optional[Callable] = None, 
                 best_idx: Optional[int] = None, 
                 best_lmda: Optional[float] = None):
        """
        Initializes the cross-validation result object.

        Additional Parameters:
        - avg_losses (np.ndarray): Average cross-validation losses.
        - cv_plot (Optional[Callable]): Function to generate cross-validation plot.
        - best_idx (Optional[int]): Index of the best-performing regularization parameter.
        - best_lmda (Optional[float]): Best regularization parameter.
        """
        super().__init__(coefs, intercept, family, gamma, gamma_intercept, beta, beta_intercepts, lasso_model, lmdas)
        self.avg_losses = avg_losses
        self.cv_plot = cv_plot
        self.best_idx = best_idx
        self.best_lmda = best_lmda

    def __repr__(self):
        base_repr = super().__repr__()
        return (f"{base_repr}, best_lmda={self.best_lmda}, "
                f"best_idx={self.best_idx}, avg_losses={self.avg_losses.shape})")
    


@jit(nopython=True, cache=True) # 性能加速器，将下面函数编码成机器码，加速
def _fit_univariate_regression_gaussian_numba(
            X: np.ndarray, 
            y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit univariate Gaussian regression for each feature in X."""
    n, p = X.shape # p是特征数量，n是数据量
    # 预先创建结果向量，固定内存空间
    beta_intercepts = np.zeros(p)
    beta_coefs = np.zeros(p)

    for j in range(p):
        xj = np.expand_dims(X[:, j], axis=1)
        xj_mean = np.mean(xj)
        y_mean = np.mean(y)
        # 最小二乘法
        sxy = np.sum(xj[:, 0] * y) - n * xj_mean * y_mean # SS_xy
        sxx = np.sum(xj[:, 0] ** 2) - n * xj_mean ** 2 # SS_xx
        slope = sxy / sxx # 斜率 SS_xy/SS_xx
        beta_intercepts[j] = y_mean - slope * xj_mean # 截距
        beta_coefs[j] = slope

    return beta_intercepts, beta_coefs


def fit_univariate_regression(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str # 新增family将模型拓展到GLM模型，做解耦，你只需输入X和y，还有family就可以了，不需要更换函数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit univariate regression model for each feature in X.

    Args:
        X: Feature matrix of shape (n, p).
        y: Target vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').

    Returns:
        Tuple containing:
            - beta_intercepts: Intercepts of the regression model.
            - beta_coefs: Coefficients of the regression model.
    """
    n, p = X.shape

    # 根据family分发任务
    if family == "gaussian":
        beta_intercepts, beta_coefs = _fit_univariate_regression_gaussian_numba(X, y)
    elif family in {"binomial", "cox"}: 
        if family == "binomial":
            glm_y = ad.glm.binomial(y)
        elif family == "cox":
            glm_y = ad.glm.cox(start=np.zeros(n), stop=y[:, 0], status=y[:, 1])

        beta_intercepts = np.zeros(p)
        beta_coefs = np.zeros(p)

        for j in range(p):
            if family == "binomial":
                X_j = np.column_stack([np.ones(n), X[:, j]]) # 把全1和当前特征拼在一起，两列
            else:
                # Cox model requires no intercept term
                X_j = np.column_stack([np.zeros(n), X[:, j]]) # 把全0和当前特征拼在一起，两列，因为cox模型不需要
            X_j = np.asfortranarray(X_j)
            glm_fit = ad.grpnet(X_j, 
                                glm_y, 
                                intercept=False, 
                                lmda_path=[0.0]) # [0.0] 意味着Lasso惩罚
            coefs = glm_fit.betas.toarray() # 

            if family == "binomial":
                beta_intercepts[j] = coefs[0][0]
           
            beta_coefs[j] = coefs[0][1]
    else:
        raise ValueError(f"Unsupported family type: {family}")

    return beta_intercepts, beta_coefs


def fit_univariate_models( 
            X: np.ndarray, 
            y: np.ndarray, 
            family: str = "gaussian" 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: # 得到单系数模型的系数和LOO拟合值
    """
    Fit univariate least squares regression for each feature in X and compute
    leave-one-out (LOO) fitted values.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').

    Returns:
        Tuple containing:
            - loo_fits: Leave-one-out fitted values.
            - beta_intercepts: Intercepts from univariate regressions.
            - beta_coefs: Slopes from univariate regressions.
    """
    beta_intercepts, beta_coefs = fit_univariate_regression(X, y, family) # 调用上一个函数
    loo_fits = fit_loo_univariate_models(X, y, family=family)["fit"] # 使用上一个函数得到的单一变量的截距和斜率做LOO估计
    # ？ 这里是不是有问题啊，其实截距和斜率是用全数据的，然后还用LOO，本来数据就已经泄露了呀？是不是应该全程LOO呢，预测截距斜率的时候就LOO
    return loo_fits, beta_intercepts, beta_coefs


def _format_unilasso_feature_matrix(X: np.ndarray, # 保安：严格检查数据矩阵X
                                    remove_zero_var: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Format and validate feature matrix for UniLasso."""

    X = np.array(X, dtype=float) # 强制转化为浮点数
    if np.any(np.isnan(X)) or np.any(np.isinf(X)): # 拦截NaN
        raise ValueError("X contains NaN or infinite values.")

    if X.ndim == 1: # 拦截维度不对的数据
        X = X.reshape(-1, 1)
    elif X.ndim != 2:
        raise ValueError("X must be a 1D or 2D NumPy array.")

    if remove_zero_var: # 剔除无意义特征，方差为0，也就是全是常数
        zero_var_idx = np.where(np.var(X, axis=0) == 0)[0]
        if len(zero_var_idx) > 0:
            warn_zero_variance(len(zero_var_idx), X.shape[1])
            X = np.delete(X, zero_var_idx, axis=1)
            if X.shape[1] == 0:
                raise ValueError("All features have zero variance.")
    else:
        zero_var_idx = None
    
    return X, zero_var_idx



def _format_unilasso_input(
            X: np.ndarray, 
            y: np.ndarray, 
            family: str, 
            lmdas: Optional[Union[float, List[float], np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray, str, Optional[np.ndarray], Optional[np.ndarray]]:
    """Format and validate input for UniLasso."""
    if family not in VALID_FAMILIES:
        raise ValueError(f"Family must be one of {VALID_FAMILIES}")
    
    X, zero_var_idx = _format_unilasso_feature_matrix(X, True)
    y = _format_y(y, family) # 格式化y

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows (samples).")

    lmdas = _format_lmdas(lmdas) # 格式化超参数

    return X, y, family, lmdas, zero_var_idx


def _format_y(
        y: Union[np.ndarray, pd.DataFrame], 
        family: str) -> np.ndarray:
    """Format and validate y based on the family."""
    if family in {"gaussian", "binomial"}:
        y = np.array(y, dtype=float).flatten() # 变成浮点数
        if family == "binomial" and not np.all(np.isin(y, [0, 1])): # 二分类数据只能是0和1
            raise ValueError("For `binomial` family, y must be binary with values 0 and 1.")
    elif family == "cox":
        if isinstance(y, (pd.DataFrame, dict)):
            if not 'time' in y.columns or not 'status' in y.columns: # Cox生存数据要包含两个维度，活了多久和标记是否去世
                raise ValueError("For `cox` family, y must be a DataFrame with columns 'time' and 'status'.")
            y = np.column_stack((y["time"], y["status"]))
        if y.shape[1] != 2:
            raise ValueError("For `cox` family, y must have two columns corresponding to time and status.")
        if not np.all(y[:, 0] >= 0):
            raise ValueError("For `cox` family, time values must be nonnegative.")
        if not np.all(np.isin(y[:, 1], [0, 1])):
            raise ValueError("For `cox` family, status values must be binary with values 0 and 1.")
    
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        raise ValueError("y contains NaN or infinite values.")
    
    return y


def _format_lmdas(lmdas: Optional[Union[float, List[float], np.ndarray]]) -> Optional[np.ndarray]: # 正则化参数
    """Format and validate lmdas."""
    if lmdas is None: # 如果不指定惩罚参数，默认最优
        return None
    if isinstance(lmdas, (float, int)):
        lmdas = [float(lmdas)] # 模型只认识列表

    if not isinstance(lmdas, list) and not isinstance(lmdas, np.ndarray): # 不允许其他
        raise ValueError("lmdas must be a nonnegative float, list of floats, or NumPy array of floats.")
    
    lmdas = np.array(lmdas, dtype=float)

    if np.any(np.isnan(lmdas)) or np.any(np.isinf(lmdas)): # 不允许Nan和无穷
        raise ValueError("Regularizers contain NaN or infinite values.")
    
    if np.any(lmdas < 0): # 不允许负数
        raise ValueError("Regularizers must be nonnegative.")

    return lmdas



def _prepare_unilasso_input(
                X: np.ndarray, 
                y: np.ndarray, 
                family: str, 
                lmdas: Optional[Union[float, List[float], np.ndarray]]
) -> Tuple[np.ndarray, 
           np.ndarray,
           np.ndarray,
           np.ndarray, 
           np.ndarray, 
           ad.glm.GlmBase64, 
           List[ad.constraint.ConstraintBase64], 
           Optional[np.ndarray]]:
    """Prepare input for UniLasso."""
    # 预处理数据
    X, y, family, lmdas, zero_var_idx = _format_unilasso_input(X, y, family, lmdas)

    # 计算单变量线性模型，计算LOO拟合值
    loo_fits, beta_intercepts, beta_coefs_fit = fit_univariate_models(X, y, family=family)
    loo_fits = np.asfortranarray(loo_fits)
    
    # 到处glm对象
    glm_family = _get_glm_family(family, y)
    
    # 施加非负约束
    # ￥取消限制
    # constraints = [ad.constraint.lower(b=np.zeros(1)) for _ in range(X.shape[1])]
    constraints = None
    return X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx




def _get_glm_family(family: str, 
                    y: np.ndarray) -> ad.glm.GlmBase64:
    """Get the appropriate GLM family."""
    if family == "gaussian":
        return ad.glm.gaussian(y)
    elif family == "binomial":
        return ad.glm.binomial(y)
    elif family == "cox":
        return ad.glm.cox(start=np.zeros(len(y)), stop=y[:, 0], status=y[:, 1])
    else:
        raise ValueError(f"Unsupported family: {family}")



def _handle_zero_variance(
            gamma_hat_fit: np.ndarray,
            beta_coefs_fit: np.ndarray,
            zero_var_idx: Optional[np.ndarray],
            cur_num_var: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Handle zero variance features."""
    if zero_var_idx is not None and len(zero_var_idx) > 0:
        total_num_var = cur_num_var + len(zero_var_idx)
        num_regs = gamma_hat_fit.shape[0]
        gamma_hat = np.zeros((num_regs, total_num_var))
        beta_coefs = np.zeros((num_regs, total_num_var))
        pos_var_idx = np.setdiff1d(np.arange(total_num_var), zero_var_idx)
        gamma_hat[:, pos_var_idx] = gamma_hat_fit
        beta_coefs[:, pos_var_idx] = beta_coefs_fit
    else:
        gamma_hat = gamma_hat_fit
        beta_coefs = beta_coefs_fit
    return gamma_hat, beta_coefs



def _print_unilasso_results(
            gamma_hat: np.ndarray, 
            lmdas: np.ndarray, 
            best_idx: Optional[int] = None
) -> None:
    """Print UniLasso results."""

    if gamma_hat.ndim == 1:
        num_selected = np.sum(gamma_hat != 0)
    else:
        num_selected = np.sum(gamma_hat != 0, axis=1)

    # check if interactive environment
    try:
        get_ipython()

        from IPython.core.display import display, HTML
        display(HTML("\n\n<b> --- UniLasso Results --- </b>"))
    except NameError:
        print("\n\n\033[1m --- UniLasso Results --- \033[0m")

    print(f"Number of Selected Features: {num_selected}")
    print(f"Regularization path (rounded to 3 decimal places): {np.round(lmdas, 3)}")
    if best_idx is not None:
        print(f"Best Regularization Parameter: {lmdas[best_idx]}")



def _format_output(lasso_model: ad.grpnet,
                   beta_coefs_fit: np.ndarray,
                   beta_intercepts: np.ndarray,
                   zero_var_idx: Optional[np.ndarray],
                   X: np.ndarray,
                   fit_intercept: bool,
                   reverse_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Format UniLasso output."""
    theta_hat = lasso_model.betas.toarray()
    theta_0 = lasso_model.intercepts

    beta_coefs_fit = beta_coefs_fit.squeeze() # 二维数组降维成一阶向量
    beta_intercepts = beta_intercepts.squeeze()

    if reverse_indices is not None:
        theta_hat = theta_hat[reverse_indices]
        theta_0 = theta_0[reverse_indices]


    gamma_hat_fit = theta_hat * beta_coefs_fit # 将单变量的权重乘Lasso的权重，恢复全局模型的权重
    gamma_hat, beta_coefs = _handle_zero_variance(gamma_hat_fit, beta_coefs_fit, zero_var_idx, X.shape[1])
    gamma_hat = gamma_hat.squeeze()
    beta_coefs = beta_coefs.squeeze()

    if fit_intercept:
        gamma_0 = theta_0 + np.sum(theta_hat * beta_intercepts, axis=1) # 全局截距
        gamma_0 = gamma_0.squeeze()
    else:
        gamma_0 = np.zeros(len(theta_0))
   
    return gamma_hat, gamma_0, beta_coefs # beta_coefs 单变量回归斜率 gamma_hat 全局斜率 gamma_0 全局截距





def _configure_lmda_min_ratio(n: int,
                              p: int) -> np.ndarray:
    """Configure lambda min ratio for UniLasso."""
    return 0.01 if n < p else 1e-4


def _check_lmda_min_ratio(lmda_min_ratio: float) -> float:
    """Check lambda min ratio for UniLasso."""
    if lmda_min_ratio <= 0:
        raise ValueError("Minimum regularization ratio must be positive.")
    if lmda_min_ratio > 1:
        raise ValueError("Minimum regularization ratio must be less than 1.")
    return lmda_min_ratio
    

def _configure_lmda_path(X: np.ndarray, 
                         y: np.ndarray,
                         family: str,
                         n_lmdas: Optional[int], 
                         lmda_min_ratio: Optional[float]) -> np.ndarray:
    """Configure the regularization path for UniLasso."""

    n, p = X.shape
    if n_lmdas is None:
        n_lmdas = 100 # 做100个不同惩罚度的模型
    
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(n, p)

    assert n_lmdas > 0, "Number of regularization parameters must be positive."
    _check_lmda_min_ratio(lmda_min_ratio)
    
    if family == "cox":
        y = y[:, 0]

    # Define function to standardize columns using n (not n-1)
    def moment_sd(z):
        return np.sqrt(np.sum((z - np.mean(z))**2) / len(z))

    X_standardized = (X - np.mean(X, axis=0)) / np.apply_along_axis(moment_sd, 0, X)
    X_standardized = np.array(X_standardized)  

    # Standardize y (centering only)
    y = y - np.mean(y)

    n = X_standardized.shape[0]
    lambda_max = np.max(np.abs(X_standardized.T @ y)) / n # 最高惩罚lambda

    lambda_path = np.exp(np.linspace(np.log(lambda_max), np.log(lambda_max * lmda_min_ratio), n_lmdas)) # 构建log(lambda)路线

    return lambda_path




# ------------------------------------------------------------------------------
# Perform cross-validation UniLasso
# ------------------------------------------------------------------------------



import numpy as np
import matplotlib.pyplot as plt

def plot(unilasso_fit) -> None:
    """
    Plots the Lasso coefficient paths as a function of the regularization parameter (lambda),
    with the number of active (nonzero) coefficients labeled at the top.

    Parameters:
    - unilasso_fit: UniLassoResult object containing fitted coefficients and lambda values.
    """
    
    assert hasattr(unilasso_fit, "coefs") and hasattr(unilasso_fit, "lmdas"), \
        "Input must have 'coefs' and 'lmdas' attributes."

    coefs, lambdas = unilasso_fit.coefs, unilasso_fit.lmdas
    if coefs.ndim == 1 or len(lambdas) == 1:
        print("Only one regularization parameter was used. No path to plot.")
        return

    plt.figure(figsize=(8, 6))
    log_lambdas = np.log(lambdas)  # Convert lambda values to log scale

    # Compute the number of nonzero coefficients at each lambda
    n_nonzero = np.sum(coefs != 0, axis=1)

    # Plot coefficient paths
    for i in range(coefs.shape[1]):  
        plt.plot(log_lambdas, coefs[:, i], lw=2)

    # Labels and formatting
    plt.xlabel(r"$\log(\lambda)$", fontsize=12)
    plt.ylabel("Coefficients", fontsize=12)
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Add secondary x-axis for the number of active coefficients
    ax1 = plt.gca()  
    ax2 = ax1.twiny()  
    ax2.set_xlim(ax1.get_xlim())  
    ax2.set_xticks(log_lambdas[::5])  
    ax2.set_xticklabels(n_nonzero[::5]) 
    ax2.set_xlabel("Number of Active Coefficients", fontsize=12)

    plt.show()


def plot_cv(cv_result: UniLassoCVResult) -> None:
    """
    Plots the cross-validation
    curve as a function of the regularization parameter (lambda).
    """
    cv_result.cv_plot()



def extract_cv(cv_result: UniLassoCVResult) -> UniLassoResult:
    """
    Extract the best coefficients and intercept from a cross-validated UniLasso result.

    Args:
        - cv_result: UniLassoCVResult object.
    
    Returns:
        - UniLassoResult object with the best coefficients and intercept.
    """

    best_coef = cv_result.coefs[cv_result.best_idx].squeeze()
    best_intercept = cv_result.intercept[cv_result.best_idx].squeeze()

    extracted_fit = UniLassoResult(
        coefs=best_coef,
        intercept=best_intercept,
        family=cv_result.family,
        gamma=best_coef,
        gamma_intercept=best_intercept,
        beta=cv_result._beta,
        beta_intercepts=cv_result._beta_intercepts,
        lasso_model=cv_result.lasso_model,
        lmdas=cv_result.lmdas
    )

    return extracted_fit

import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

class CustomSoftPenaltyCVLasso:
    """一个伪装成 ad.cv_grpnet 的自定义交叉验证引擎"""
    
    def __init__(self, X, y, lmdas, n_folds=5, alpha_penalty=50.0, seed=None):
        self.X = X
        self.y = y
        self.lmdas = lmdas
        self.n_folds = n_folds
        self.alpha_penalty = alpha_penalty
        self.seed = seed
        
        # 准备向外暴露的“接口暗号”（必须和 C++ 引擎的属性名一模一样）
        self.avg_losses = None
        self.best_idx = None
        
        # 立即开始轰轰烈烈的交叉验证！
        self._run_cv()

    def _run_cv(self):
        # 1. 准备 K 折交叉验证切分器
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        # 准备一个矩阵，记录每一次考试的成绩：形状 (5折, 100个lambda)
        fold_losses = np.zeros((self.n_folds, len(self.lmdas)))
        
        # 2. 开始切分数据并循环考试
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]
            
            # 使用我们上一节课写的引擎，在 80% 的数据上训练
            fold_model = CustomSoftPenaltyLasso(
                X=X_train, y=y_train, lmda_path=self.lmdas, alpha_penalty=self.alpha_penalty
            )
            
            # 拿出训练好的 100 套权重
            dense_betas = fold_model.betas.toarray() 
            
            # 在剩下的 20% 测试数据上考试，算 MSE
            for i, lmda in enumerate(self.lmdas):
                y_hat = X_val @ dense_betas[i] + fold_model.intercepts[i]
                mse = np.mean((y_val - y_hat) ** 2)
                fold_losses[fold_idx, i] = mse
                
        # 3. 考试结束！计算每个 lambda 的平均成绩
        self.avg_losses = np.mean(fold_losses, axis=0)
        
        # 4. 找到成绩最好（误差最小）的那个 lambda 的索引！
        self.best_idx = np.argmin(self.avg_losses)

    def fit(self, X, glm=None, groups=None, intercept=True, constraints=None):
        """
        这个方法极其重要！
        下游代码会调用 `cv_lasso.fit(...)`，要求我们在 100% 的全量数据上重新训练一次。
        这里我们接收一堆 C++ 引擎才会用的参数（比如 glm, constraints），但我们在内部直接忽略它们。
        """
        # 召唤主力军，在 100% 的数据上重新拟合！
        refitted_model = CustomSoftPenaltyLasso(
            X=self.X, y=self.y, lmda_path=self.lmdas, alpha_penalty=self.alpha_penalty
        )
        return refitted_model

    def plot_loss(self):
        """伪装 C++ 引擎自带的画图函数"""
        plt.figure(figsize=(8, 6))
        plt.plot(np.log(self.lmdas), self.avg_losses, marker='o', color='blue')
        plt.axvline(np.log(self.lmdas[self.best_idx]), color='red', linestyle='--', label='Best Lambda')
        plt.xlabel('log(Lambda)')
        plt.ylabel('Cross-Validation MSE')
        plt.title('Custom Soft Penalty CV Loss Curve')
        plt.legend()
        plt.show()

def cv_unilasso( # 封装所有东西，融合所有东西的类
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            n_folds: int = 5,
            lmda_min_ratio: float = None,
            verbose: bool = False,
            seed: Optional[int] = None
) ->  UniLassoCVResult:
    """
    Perform cross-validation UniLasso.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        n_folds: Number of cross-validation folds.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter.
        verbose: Whether to print results.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary containing UniLasso results.
    """
    if lmda_min_ratio is None:
        lmda_min_ratio = _configure_lmda_min_ratio(X.shape[0], X.shape[1])

    assert n_folds > 1, "Number of folds must be greater than 1."
    _check_lmda_min_ratio(lmda_min_ratio)
    
    # 得到单变量模型的食材
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, _, zero_var_idx = _prepare_unilasso_input(X, y, family, None)
    fit_intercept = False if family == "cox" else True

    
    lmdas = _configure_lmda_path(X=loo_fits, y=y, family=family, 
                                 n_lmdas=100, lmda_min_ratio=lmda_min_ratio)
    
    
    # cv_lasso = ad.cv_grpnet(
    #     X=loo_fits, # LOO拟合值
    #     glm=glm_family,
    #     seed=seed,
    #     n_folds=n_folds,
    #     groups=None,
    #     min_ratio=lmda_min_ratio,
    #     intercept=fit_intercept,
    #     constraints=constraints,
    #     tol=1e-7
    # )
    
    cv_lasso = CustomSoftPenaltyCVLasso(
        X=loo_fits, 
        y=y, 
        lmdas=lmdas, 
        n_folds=n_folds, 
        alpha_penalty=50.0, # 沼泽地的深度
        seed=seed
    )

    # refit lasso along a regularization path that stops at the best chosen lambda
    lasso_model = cv_lasso.fit( # 寻找最优参数，然后用全数据训练一次
        X=loo_fits,
        glm=glm_family,
        groups=None,
        intercept=fit_intercept,
        constraints=constraints,
    )

    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model, # 整理全局模型的输出
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept)

    

    cv_plot = cv_lasso.plot_loss
    if verbose:
        _print_unilasso_results(gamma_hat, cv_lasso.lmdas, int(cv_lasso.best_idx))
        cv_plot()
    

    unilasso_result = UniLassoCVResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=cv_lasso.lmdas,
        avg_losses=cv_lasso.avg_losses,
        cv_plot=cv_plot,
        best_idx=int(cv_lasso.best_idx),
        best_lmda=cv_lasso.lmdas[cv_lasso.best_idx]
    ) # 返回一个结果，甚至可以画图

    return unilasso_result


# ------------------------------------------------------------------------------
# Fit UniLasso for a specified regularization path
# ------------------------------------------------------------------------------
def fit_unilasso( # 如果没有lambda传入，直接使用最优训练误差，榨干每一点数据
            X: np.ndarray,
            y: np.ndarray,
            family: str = "gaussian",
            lmdas: Optional[Union[float, List[float], np.ndarray]] = None,
            n_lmdas: Optional[int] = 100,
            lmda_min_ratio: Optional[float] = 1e-2,
            verbose: bool = False
) -> UniLassoResult:
    """
    Perform UniLasso with specified regularization parameters.

    Args:
        X: Feature matrix of shape (n, p).
        y: Response vector of shape (n,).
        family: Family of the response variable ('gaussian', 'binomial', 'cox').
        lmdas: Lasso regularization parameter(s).
        n_lmdas: Number of regularization parameters to use if `lmdas` is None.
        lmda_min_ratio: Minimum ratio of the largest to smallest regularization parameter. 
        verbose: Whether to print results.

    Returns:
        Dictionary containing UniLasso results.
    """
    # 输出单变量模型
    X, y, loo_fits, beta_intercepts, beta_coefs_fit, glm_family, constraints, lmdas, zero_var_idx = _prepare_unilasso_input(X, y, family, lmdas)

    fit_intercept = False if family == "cox" else True

    # 拟合lasso模型，但是没有做cv，只跑一次全量数据，速度会快很多
    # lasso_model = ad.grpnet(
    #     X=loo_fits,
    #     glm=glm_family,
    #     groups=None,
    #     intercept=fit_intercept,
    #     lmda_path=lmdas, # Regularization path, if unspecified, will be generated 如果没有制定lambda，会自动生成一个lambda
    #     constraints=constraints,
    #     lmda_path_size=n_lmdas,
    #     min_ratio=lmda_min_ratio,
    #     tol=1e-7
    # )
    if lmdas is None:
        # 调用系统自带的路线生成器
        lmdas = _configure_lmda_path(X=loo_fits, 
                                     y=y, 
                                     family=family, 
                                     n_lmdas=n_lmdas, 
                                     lmda_min_ratio=lmda_min_ratio)
    lasso_model = CustomSoftPenaltyLasso(
        X=loo_fits, 
        y=y, 
        lmda_path=lmdas, 
        alpha_penalty=50.0 # 沼泽地的深度，你可以自己调
    )

    glm_lmdas = np.array(lasso_model.lmdas)

    if lmdas is not None: # 如果没有制定lambda，会自动跑一百个lambda，然后选择最优的
        if not np.all(np.isin(lmdas, glm_lmdas)):
            removed_lmdas = np.setdiff1d(lmdas, glm_lmdas)
            removed_lmdas = np.round(removed_lmdas, 3)
            warn_removed_lmdas(removed_lmdas)

        matching_idx = np.where(np.isin(lmdas, glm_lmdas))[0]
        lmdas = lmdas[matching_idx]
    else:
        lmdas = glm_lmdas

    if len(lmdas) == 0:
        raise ValueError("No regularization strengths remain after removing invalid values")

    reverse_indices = np.arange(len(glm_lmdas))
    reverse_indices = reverse_indices[::-1]


    gamma_hat, gamma_0, beta_coefs = _format_output(lasso_model,
                                                    beta_coefs_fit,
                                                    beta_intercepts,
                                                    zero_var_idx,
                                                    X,
                                                    fit_intercept,
                                                    reverse_indices)

    if verbose:
        _print_unilasso_results(gamma_hat, lmdas)

    unilasso_result = UniLassoResult(
        coefs=gamma_hat,
        intercept=gamma_0,
        family=family,
        gamma=gamma_hat,
        gamma_intercept=gamma_0,
        beta=beta_coefs,
        beta_intercepts=beta_intercepts,
        lasso_model=lasso_model,
        lmdas=lmdas
    )

    return unilasso_result


def predict(result: UniLassoResult,
            X: np.ndarray, 
            lmda_idx: Optional[int] = None) -> np.ndarray:
    """
    Predict response variable using UniLasso model.

    Args:
        result: UniLasso result object.
        X: Feature matrix of shape (n, p).
        lmda_idx: Index of the regularization parameter to use for prediction.

    Returns:
        Predicted response variable.
    """

    if not type(result) == UniLassoResult:
        raise ValueError("`result` must be a UniLassoResult object.")
    
    if len(result.coefs.shape) == 1:
        result.coefs = np.expand_dims(result.coefs, axis=0)
    assert result.coefs.shape[1] == X.shape[1], "Feature matrix must have the same number of columns as the fitted model."

    X, _ = _format_unilasso_feature_matrix(X, remove_zero_var=False)
    
    if lmda_idx is not None:
        assert lmda_idx >= 0 and lmda_idx < len(result.lmdas), "Invalid regularization parameter index."
        y_hat = X @ result.coefs[lmda_idx] + result.intercept[lmda_idx]
    else:
        y_hat = X @ result.coefs.T + result.intercept 

    y_hat = y_hat.squeeze()
          
    return y_hat
