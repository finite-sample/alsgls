## Usage

```python
from alsgls import ALSGLS, ALSGLSSystem, simulate_sur

Xs_tr, Y_tr, Xs_te, Y_te = simulate_sur(N_tr=240, N_te=120, K=60, p=3, k=4)

# Scikit-learn style estimator
est = ALSGLS(rank="auto", max_sweeps=12)
est.fit(Xs_tr, Y_tr)
test_score = est.score(Xs_te, Y_te)  # negative test NLL per observation

# Statsmodels-style system interface
system = {f"eq{j}": (Y_tr[:, j], Xs_tr[j]) for j in range(Y_tr.shape[1])}
sys_model = ALSGLSSystem(system, rank="auto")
sys_results = sys_model.fit()
params = sys_results.params_as_series()  # pandas optional
```