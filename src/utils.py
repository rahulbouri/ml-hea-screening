# import numpy as np
# from scipy.stats import norm
# from tqdm import tqdm  # progress bar

# # DETERMINING the mean and standard deviation using bootstrap sampling with 1000 bootstrap samples
# def bootstrap_estimator(model, X_train, y_train, X_test, n_iter=1000):
#     '''
#     Function to estimate the mean and standard deviation using bootstrap sampling.

#     Parameters:
#     model       = best model with hyperparameters
#     X_train     = training features
#     y_train     = training target
#     X_test      = test features
#     n_iter      = number of bootstrap samples

#     Returns:
#     Mean and standard deviation of predictions over the bootstrap samples.
#     '''
#     n_train = X_train.shape[0]
#     n_test = len(X_test)
#     print(f"[bootstrap_estimator] Starting with n_train={n_train}, n_test={n_test}, n_iter={n_iter}")

#     bootstrap_preds = np.zeros((n_test, n_iter))
#     index = np.arange(n_train)

#     # Set seed to repeat the bootstrapping
#     np.random.seed(20)

#     # Wrap with tqdm to track progress
#     for i in tqdm(range(n_iter), desc="Bootstrapping", unit="iter"):
#         if i % max(1, n_iter // 10) == 0:
#             print(f"[bootstrap_estimator] Iteration {i}/{n_iter}")

#         # Sample with replacement
#         index_sampled = np.random.choice(index, size=n_train, replace=True)
#         X_train_sample = X_train[index_sampled, :]
#         y_train_sample = y_train[index_sampled]

#         # Fit and predict
#         model.fit(X_train_sample, y_train_sample)
#         bootstrap_preds[:, i] = model.predict(X_test)

#     mu = bootstrap_preds.mean(axis=1)
#     sigma = bootstrap_preds.std(axis=1)
#     print("[bootstrap_estimator] Completed bootstrap sampling")
#     return mu, sigma


# # Expected Improvement Calculation
# def Expected_Improvement(X_test, X_train_all, y_train_all, model, xi=0.01):
#     '''
#     Calculate the Expected Improvement (EI) acquisition function for Bayesian Optimization.

#     Parameters:
#     X_test      = candidate points to evaluate
#     X_train_all = complete training feature set
#     y_train_all = complete training target set
#     model       = regression model (e.g., GBR)
#     xi          = exploration-exploitation trade-off parameter

#     Returns:
#     Tuple of (EI, mean predictions, standard deviation)
#     '''
#     n_candidates = len(X_test)
#     print(f"[Expected_Improvement] Starting for {n_candidates} candidates with xi={xi}")

#     mu_x, sigma_x = bootstrap_estimator(model, X_train_all, y_train_all, X_test, n_iter=1000)
#     print(f"[Expected_Improvement] Received mu_x.shape={mu_x.shape}, sigma_x.shape={sigma_x.shape}")

#     mu_max = np.max(y_train_all)
#     print(f"[Expected_Improvement] Maximum observed y_train_all: mu_max={mu_max}")

#     diff = mu_x - mu_max - xi
#     z = diff / sigma_x
#     ei = diff * norm.cdf(z) + sigma_x * norm.pdf(z)
#     ei[sigma_x == 0.0] = 0.0  # avoid division by zero

#     print("[Expected_Improvement] Completed EI calculation")
#     return ei, mu_x, sigma_x


import numpy as np
from scipy.stats import norm
from tqdm import tqdm  # progress bar

def bootstrap_estimator(model, X_train, y_train, X_test, n_iter=1000):
    """
    Estimate mean and stddev of model predictions via bootstrapping.

    Parameters
    ----------
    model    : scikit-learn regressor (already hyper-tuned)
    X_train  : array‑like, shape (n_train, n_features)
    y_train  : array‑like, shape (n_train,)
    X_test   : array‑like, shape (n_test, n_features)
    n_iter   : int, number of bootstrap samples

    Returns
    -------
    mu       : ndarray, shape (n_test,)  — mean prediction
    sigma    : ndarray, shape (n_test,)  — stddev of predictions
    """
    n_train = X_train.shape[0]
    n_test = len(X_test)
    print(f"[bootstrap_estimator] n_train={n_train}, n_test={n_test}, n_iter={n_iter}")

    # keep a small scratch array per batch
    bootstrap_preds = np.zeros((n_test, n_iter), dtype=float)
    idx = np.arange(n_train)

    # fixed seed for reproducibility
    rng = np.random.default_rng(20)

    for i in tqdm(range(n_iter), desc="Bootstrapping", unit="iter"):
        # sample with replacement
        sample_idx = rng.choice(idx, size=n_train, replace=True)
        Xs = X_train[sample_idx]
        ys = y_train[sample_idx]

        model.fit(Xs, ys)
        bootstrap_preds[:, i] = model.predict(X_test)

    mu = bootstrap_preds.mean(axis=1)
    sigma = bootstrap_preds.std(axis=1, ddof=1)
    print("[bootstrap_estimator] done")
    return mu, sigma


def Expected_Improvement(X_test, X_train_all, y_train_all, model, xi=0.01, batch_size=100_000):
    """
    Batched EI calculation so you don’t OOM on large X_test.

    Parameters
    ----------
    X_test      : array‑like, shape (n_candidates, n_features)
    X_train_all : array‑like, shape (n_train, n_features)
    y_train_all : array‑like, shape (n_train,)
    model       : scikit‑learn regressor
    xi          : float, exploration parameter
    batch_size  : int, number of candidates per batch

    Returns
    -------
    ei       : ndarray, shape (n_candidates,)
    mu_x     : ndarray, shape (n_candidates,)
    sigma_x  : ndarray, shape (n_candidates,)
    """
    n_candidates = len(X_test)
    print(f"[Expected_Improvement] {n_candidates} candidates, xi={xi}, batch_size={batch_size}")

    # Preallocate outputs
    ei      = np.zeros(n_candidates, dtype=float)
    mu_x    = np.zeros(n_candidates, dtype=float)
    sigma_x = np.zeros(n_candidates, dtype=float)

    mu_max = np.max(y_train_all)

    # Process in batches
    for start in range(0, n_candidates, batch_size):
        end = min(start + batch_size, n_candidates)
        X_chunk = X_test[start:end]

        # bootstrap on this chunk
        mu_chunk, sigma_chunk = bootstrap_estimator(
            model, X_train_all, y_train_all, X_chunk, n_iter=1000
        )

        # Compute EI for the chunk
        diff = mu_chunk - mu_max - xi
        z    = diff / sigma_chunk
        ei_chunk = diff * norm.cdf(z) + sigma_chunk * norm.pdf(z)
        ei_chunk[sigma_chunk == 0] = 0.0

        # Store results
        mu_x[start:end]    = mu_chunk
        sigma_x[start:end] = sigma_chunk
        ei[start:end]      = ei_chunk

    print("[Expected_Improvement] completed")
    return ei, mu_x, sigma_x
