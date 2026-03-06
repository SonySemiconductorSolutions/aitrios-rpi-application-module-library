#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np


def residual_var(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the variance of the residuals from regressing y on x (via least squares).

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        Variance of the residuals.
    """
    x_mean: float = np.mean(x)
    y_mean: float = np.mean(y)

    # Calculate slope (beta) and intercept (alpha) for the best-fit line
    slope: float = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) * (x - x_mean))
    intercept: float = y_mean - slope * x_mean

    # Predict y values using the regression line
    y_pred: np.ndarray = slope * x + intercept
    residual: np.ndarray = y - y_pred

    return np.var(residual)


def corr_and_residual_var(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compute the correlation coefficient and residual variance between two arrays.

    Args:
        A: First input array.
        B: Second input array.

    Returns:
        Dictionary containing:
            - 'corr': Pearson correlation coefficient between A and B.
            - 'residual_var': Variance of the residuals when regressing B on A.
    """
    val = {
        "corr": np.corrcoef(A, B)[0, 1],
        "residual_var": residual_var(A, B),
    }
    return val
