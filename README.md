# Causal ML Meta-Learner Benchmarking Suite

A modular, from-scratch implementation and stress-testing framework for causal machine learning meta-learners (**S-Learner**, **T-Learner**, and **X-Learner**). Built to evaluate estimator resilience against sample size constraints, regularization bias, and severe positivity strain (overlap violations) using ground-truth PEHE metrics.

---

## 📂 Project Architecture

* **`generate_synthetic_data()`**: A semi-synthetic Data Generating Process (DGP) featuring controllable overlap/propensity strain ($\gamma$) and customizable treatment effect heterogeneity (HTE) structures (linear vs. complex non-linear interactions).
* **`train_s_learner()`**: Trains a single model across features and treatment assignment $[X, W]$, illustrating the regularization shrinkage trap.
* **`train_t_learner()`**: Trains independent models for treated and control arms ($\hat{\mu}_1, \hat{\mu}_0$), bypassing shrinkage but vulnerable to sample-split variance and extrapolation error.
* **`train_x_learner()`**: A sophisticated 4-stage meta-learner utilizing pseudo-outcome residual imputation and propensity score blending ($e(X)$) to handle severe treatment arm imbalance.
* **Benchmarking Loop**: Evaluates model performance across stress-test scenarios using **PEHE (Precision in Estimation of Heterogeneous Effects)**.

---

## 📊 Key Methodological Findings

Across empirical stress-tests comparing sample sizes ($N$) and positivity strain ($\gamma$):
1. **X-Learner Dominance under Imbalance:** Consistently achieves the lowest PEHE error across unbalanced and confounded regimes due to its residual-based propensity weighting mechanism.
2. **T-Learner Extrapolation Trap:** Vulnerable to severe error spikes when overlap breaks down ($\gamma = 3.5$), forcing models to extrapolate into regions lacking data density.
3. **The Regularization/Variance Trade-off:** While S-Learners suffer from shrinkage bias, scaling to large $N$ with complex non-linear treatment effects allows unregularized architectures (like the T-Learner) to overcome shrinkage penalties.

---

## 🚀 Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/csmedley1004/causal-ml-meta-learners.git](https://github.com/csmedley1004/causal-ml-meta-learners.git)
