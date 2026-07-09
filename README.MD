# Formative 3: EM Algorithm, Bayesian Probability & Gradient Descent

Group project for ALU — Probability Distributions, Bayesian Probability, and Gradient Descent.

## Team

| Member | Role / Responsibility |
|---|---|
| **Nzabinesha Merci** | EM Algorithm (Part 1) — data loading, E-step, initialization, tracking table, live classification demo |
| **Ivan** | Bayesian Probability (Part 2) — keyword selection, Bayes' Theorem implementation |
| **Dorian** | Gradient Descent Manual (Part 3) — chain-rule derivation, iteration calculations |
| **Belyse** | Gradient Descent Code (Part 4) — SciPy derivative function, Matplotlib plots |

All four members contributed one full gradient descent iteration each in Part 3, and all four are prepared to speak to every part of the project during the live presentation.

---

## Part 1 — Expectation-Maximization (EM) Algorithm

**Dataset:** GaltonFamilies.csv (father heights, deduplicated one-per-family, pooled with all children's heights). Pooled dataset: 205 unique fathers + 934 children = **1,139 unlabeled data points**.

**Objective:** treat the pooled heights as an unlabeled mixture of two Gaussian distributions and recover the two underlying groups without using the true father/child labels during fitting.

### Should you just split the data at the global mean?

**No.** The global mean of the pooled dataset is 67.209 inches. Splitting there gives two piles averaging 64.26 and 70.22 — but **50.2% of all 1,139 points (572 of them) fall within just 3 inches of that boundary.** Over half the dataset sits close enough to the cutoff that a hard split is essentially guessing for most of it. EM instead assigns every point a soft, probabilistic responsibility for each group, which honestly reflects that ambiguity instead of hiding it.

### Tracking table (real output)

| Iteration | μ1 (Children) | μ2 (Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 (Init) | 64.2299 | 70.1819 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3136.4676 |
| 1 | 65.1207 | 69.2760 | 8.7032 | 8.1149 | 0.4976 | 0.5024 | -3056.6103 |
| 2 | 65.0591 | 69.3312 | 8.5547 | 7.7733 | 0.4969 | 0.5031 | -3055.5576 |

Initialization is fully unsupervised: the pooled data is sorted and split into a lower/upper half by position (not by true label), and each half's mean seeds μ1/μ2. Log-likelihood increases every iteration, as EM guarantees, and the increase shrinks over time — the model is converging.

### Live classification demo

`classify_height(test_height, ...)` reuses the E-step formula on a single new height and returns P(Child) and P(Pro). Example: at 71.0 inches → P(Child) = 0.1252, P(Pro) = 0.8748. During the presentation, this is re-run live with whatever height the coach provides.

### Visuals included in the notebook

- Histogram of the pooled data with both fitted Gaussian curves overlaid, shaded where they overlap
- μ1/μ2 and log-likelihood evolution across iterations (pulled directly from the executed `table_rows` variable, not hardcoded)
- Bar chart comparing the naive mean-split piles against the EM-recovered component means

---

## Part 2 — Bayesian Probability (IMDb Movie Reviews)

**Dataset:** IMDb Movie Reviews dataset, as provided for the assignment.

**We calculate only** P(Positive | Keyword), as required.

**Keywords selected:**
- Positive: `excellent`, `amazing`, `great`, `love`
- Negative (identified but not computed): `terrible`, `awful`, `boring`, `worst`

**Justification:** these words carry strong emotional charge and are commonly used to express clear satisfaction or disappointment with a movie, making them reliable sentiment signals.

### Results (real output)

| Keyword | Prior | Posterior |
|---|---|---|
| excellent | 0.50267 | 0.8595 |
| amazing | 0.50267 | 0.7716 |
| love | 0.50267 | 0.68284 |
| great | 0.50267 | 0.65253 |

*(Full likelihood/marginal values for each keyword are printed in the notebook output.)*

**Interpretation:** "excellent" is the strongest positive-sentiment indicator of the four, followed by "amazing," then "love," then "great." "Great" is the weakest indicator because it's common enough to appear across many negative reviews too (e.g. sarcastic or mixed reviews), pulling its posterior closer to the 0.5 baseline.

### Implementation notes

- Pure Python (no ML libraries), as required
- Modular: `load_reviews()`, `compute_prior()`, and a single `bayes_posterior()` function reused across all four keywords — no repeated counting logic per keyword

---

## Part 3 — Gradient Descent (Manual Calculation)

Handwritten in a separate submitted PDF, one full iteration per group member (4 members → 4 iterations), each showing the chain rule explicitly for both ∂J/∂m and ∂J/∂b.

**Setup:** `y_hat = Xm + b`, X = [[1,3],[4,10]], initial m = [-1, 2], initial b = [1, 1], y = [5, 6], learning rate α = 0.01.

**Loss trend across the 4 iterations:** 61.00 → 6.50 → 2.50 → 2.16 — steadily decreasing, confirming gradient descent is moving m and b in a direction that reduces error. The largest correction happens in the first step; each subsequent step refines further as the parameters approach a better fit.

---

## Part 4 — Gradient Descent (Code)

Converts the Part 3 manual work into Python, matching every number exactly.

- `predict()`, `mse_cost()` — the model and cost function
- `scipy_gradient()` — generic derivative helper using `scipy.optimize.approx_fprime`, accepts any cost function and parameter vector
- `analytical_gradient()` — the chain-rule gradient from Part 3, used to cross-check the SciPy result at every iteration
- `gradient_descent_step()` — one reusable function performing a single update (forward pass → gradient → parameter update); the training loop calls this function once per iteration instead of repeating the math inline

**Final result after 4 iterations:** m = [-1.3637, 1.1191], b = [1.0568, 0.8949], final MSE = 2.099581 — matching the handwritten Part 3 calculations exactly.

**Plots included:**
1. Evolution of m and b over iterations
2. Error (MSE) over iterations

---

## Repository structure

```
├── father_notebook_final.ipynb    # Parts 1, 2, and 4 (EM, Bayesian, Gradient Descent code)
├── GaltonFamilies.csv             # Height dataset used in Part 1
├── IMDB Dataset.csv               # Review dataset used in Part 2 (provided separately, not tracked in repo)
├── Part_3_Manual_Calculations.pdf # Handwritten gradient descent derivations            
└── README.md                      # This file
```

## How to run

1. Open `father_notebook_final.ipynb` in Google Colab
2. Run all cells top to bottom
3. For Part 1, `GaltonFamilies.csv` must be present in the working directory
4. For Part 2, upload `IMDB Dataset.csv` when prompted by the file-upload cell


