# Flam-assignment

## 1. Introduction
Estimate the unknown parameters **θ**, **M**, and **X** in a nonlinear model using **L1 distance minimization** between observed and predicted data points.

---

## 2. Model Equations

\[
x(t) = t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X
\]

\[
y(t) = 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\]

Where:
- \( t \): parameter uniformly sampled between **6** and **60**
- \( \theta \): rotation angle (degrees/radians)
- \( M \): exponential growth/decay rate
- \( X \): horizontal shift

---

## 3. Approach

### 1. **Data Loading**
Loaded the dataset **`xy_data.csv`**, containing **1500** (x, y) points.

### 2. **Model Definition**
Defined a function `model_xy(t, θ, M, X)` to generate predicted x(t) and y(t).

### 3. **Loss Function (L1 Distance)**
To ensure robustness to outliers, the **L1 norm** was used:

\[
L_1 = \sum_i (|x_i - \hat{x_i}| + |y_i - \hat{y_i}|)
\]

### 4. **Initial Grid Search**
Performed a coarse search across:
- θ ∈ [1°, 49°]  
- M ∈ [-0.045, 0.045]  
- X initialized from mean offset

### 5. **Optimization**
Refined the parameters using the **L-BFGS-B** optimization algorithm  
(`scipy.optimize.minimize`) with proper bounds.

### 6. **Preprocessing**
Sorted data by **x-coordinate** for consistent model alignment.

### 7. **Evaluation**
Computed **total** and **average** L1 losses after optimization.

---

## 4. Results

| Parameter | Symbol | Estimated Value |
|------------|----------|----------------:|
| Rotation Angle | \( \theta \) | 30.0436° |
| Exponential Rate | \( M \) | 0.029991 |
| X-Offset | \( X \) | 55.0155 |

**Total L1 Loss:** 453.4369  
**Average L1 Loss:** 0.3023  

---

## 5. Final Estimated Model

\[
\begin{aligned}
x(t) &= t\cos(0.5244) - e^{0.029991|t|}\sin(0.3t)\sin(0.5244) + 55.0155 \\
y(t) &= 42 + t\sin(0.5244) + e^{0.029991|t|}\sin(0.3t)\cos(0.5244)
\end{aligned}
\]

*(Note: 0.5244 radians ≈ 30.0436°)*

---

## 6. Summary
By leveraging **L1-based error minimization** and **gradient-based refinement**, the model accurately captures nonlinear behavior in the observed data.  
The low **average L1 loss (≈0.3023)** demonstrates excellent parameter recovery and model robustness.

---

## 7. Dependencies
- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib (for visualization)

Install them using:
```bash
pip install numpy scipy matplotlib
