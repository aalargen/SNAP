import numpy as np
from scipy import optimize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from snap.ridge_gcv_mod import RidgeCVMod

import torch
from tqdm import tqdm

import jax
import jax.numpy as jnp
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
jax.config.update("jax_enable_x64", True)


@jax.jit
def denom_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    return p*eigs + kappa


@jax.jit
def delta_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return (eigs / denom**2).sum()


@jax.jit
def gamma_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return (p*eigs**2 / denom**2).sum()


@jax.jit
def eff_lambda(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    eff_reg = kappa - kappa * (eigs/denom).sum()
    return eff_reg


# Definition of kappa and its derivatives
@jax.jit
def kappa_fn(kappa, *args):
    (p, reg, eigs, weights_sq) = args
    kappa = jnp.abs(kappa)

    denom = denom_fn(kappa, *args)
    return kappa - reg - kappa * np.sum(eigs/denom)


kappa_prime = jax.jit(jax.grad(kappa_fn, argnums=0))
kappa_pprime = jax.jit(jax.grad(kappa_prime, argnums=0))


def solve_kappa_gamma(pvals, reg, eigs, weights_sq):

    eigs = np.abs(eigs)

    fun, fprime, fprime2 = kappa_fn, kappa_prime, kappa_pprime

    if type(reg) not in [list, np.ndarray]:
        reg = [reg] * len(pvals)
    reg = np.array(reg)

    kappa_vals = np.zeros(len(pvals))
    gamma_vals = np.zeros(len(pvals))
    eff_regs = np.zeros(len(pvals))
    for i, (p, lamb) in enumerate(zip(pvals, reg)):
        args = (p, lamb*p, eigs, weights_sq)

        kappa_0 = lamb + np.sum(eigs)  # When p = 0
        kappa_1 = lamb                 # When p is infty

        kappa_vals[i] = optimize.root_scalar(fun,
                                             fprime=fprime,
                                             fprime2=fprime2,
                                             args=args,
                                             x0=kappa_0,
                                             x1=kappa_1,
                                             method='newton',
                                             xtol=1e-12, maxiter=200).root

        gamma_vals[i] = gamma_fn(kappa_vals[i], *args)
        eff_regs[i] = eff_lambda(kappa_vals[i], *args) / p

    kappa_vals = np.abs(np.nan_to_num(kappa_vals))
    gamma_vals = np.nan_to_num(gamma_vals)
    eff_regs = np.nan_to_num(eff_regs)
    eff_regs = eff_regs + 1e-14

    return np.array(kappa_vals), np.array(gamma_vals), np.array(eff_regs)


def gen_error_theory(eigs, weights, reg, pvals=None):

    # Number of classes
    if len(weights.shape) == 1:
        weights = weights.reshape(-1, 1)
    C = weights.shape[-1]

    # Sample size for theory
    P = eigs.shape[0]
    if pvals is None:
        pvals = [int(.6*P), int(.8*P)]

    # Absolute value of eigs improves numerical stability
    eigs = np.abs(eigs)
    weights_sq = (weights**2).sum(-1)
    alignment = weights**2 / weights_sq.sum()

    # Solve for self-consistent equation
    kappa, gamma, eff_regs = solve_kappa_gamma(pvals, reg, eigs, weights_sq)

    # Calculate generalization and training error
    prefactor_gen = kappa ** 2 / (1 - gamma)
    prefactor_tr = eff_regs**2 / kappa**2

    errors = {'pvals_theory': pvals,
              'kappa': kappa,
              'gamma': gamma,
              'eff_regs': eff_regs,
              'E_i': np.zeros((len(pvals), len(eigs))),
              'gen_theory': np.zeros((len(pvals), C)),
              'tr_theory': np.zeros((len(pvals), C)),
              'radius_theory': np.zeros((len(pvals))),
              'dimension_theory': np.zeros((len(pvals))),
              'error_modes_theory': np.zeros((len(pvals), P, C)),
              }

    for i, p in enumerate(pvals):
        E_i = prefactor_gen[i] * (1 / (p*eigs + kappa[i])**2)
        error_mode = E_i[:, None] * alignment 

        for j in range(C):
            # Normalize by L2 norm of target
            gen_err = (error_mode[:, j]).sum() # total error per voxel
            tr_err = prefactor_tr[i] * gen_err

            errors['gen_theory'][i, j] = gen_err
            errors['tr_theory'][i, j] = tr_err
            
        errors['E_i'][i] = E_i
        errors['error_modes_theory'][i] = error_mode
        
        # find radius and dimension
        sum_sq_err_modes = np.square(error_mode).sum()
        sum_err_modes_sq = np.square(errors['gen_theory'][i].sum(-1))
        radius = np.sqrt(sum_sq_err_modes)
        dimension = sum_err_modes_sq/sum_sq_err_modes
        
        errors['radius_theory'][i] = radius
        errors['dimension_theory'][i] = dimension

    return errors


@torch.no_grad()
def regression(feat, y, pvals=None, cent=False, num_trials=3, reg=None, **kwargs):

    P, N = feat.shape
    C = y.shape[-1]

    # To extract the learning curve divide samples into 10
    if pvals is None:
        pvals = [int(.6*P), int(.8*P)]

    if cent:
        y -= y.mean(0, keepdim=True)
        feat -= feat.mean(0, keepdim=True)

    errors = {'pvals': pvals,
              'P': P,
              'N': N,
              'C': C,
              'cent': cent,
              'reg': np.zeros(len(pvals)), 

              'gen_errs': np.zeros((num_trials, len(pvals), C)),
              'tr_errs': np.zeros((num_trials, len(pvals), C)),
              'test_errs': np.zeros((num_trials, len(pvals), C)),

              'r2_gen': np.zeros((num_trials, len(pvals), C)),
              'r2_tr': np.zeros((num_trials, len(pvals), C)),
              'r2_test': np.zeros((num_trials, len(pvals), C)),

              'pearson_tr': np.zeros((num_trials, len(pvals), C)),
              'pearson_test': np.zeros((num_trials, len(pvals), C)),
              'pearson_gen': np.zeros((num_trials, len(pvals), C)),

              'gen_norm': np.zeros((num_trials, len(pvals), C)),
              'tr_norm': np.zeros((num_trials, len(pvals), C)),
              'test_norm': np.zeros((num_trials, len(pvals), C)),
              }
    
    if reg is None:
        alphas = np.logspace(-15, 10, 26).tolist() # do 51 steps to get between too
    elif isinstance(reg, (int, float)):
        alphas = [reg]
    else:
        alphas = reg


    for i, p in enumerate(pvals):
        best_alpha = None
        for j in range(num_trials):

            idx, idx_test = train_test_split(np.arange(0, P, 1), train_size=p)
            assert len(set(idx)) == p
            assert len(set(idx_test)) == P - p

            y_tr = y[idx]
            y_test = y[idx_test]
            feat_tr = feat[idx]

            #if first trial, use RidgeCV to get an alpha
            if best_alpha is None:
                ridge_cv = RidgeCVMod(alphas=alphas, store_cv_values=False,
                                      alpha_per_target=False, scoring='pearson_r',
                                      fit_intercept=False)
                ridge_cv.fit(np.array(feat_tr), np.array(y_tr))
                best_alpha = ridge_cv.alpha_
                print(f'\n N: {N}, p: {p}, Best Alpha: {best_alpha}')
                errors['reg'][i] = best_alpha/p

                y_hat = torch.from_numpy(ridge_cv.predict(np.array(feat)))
                del ridge_cv

            else:
                #sklearn ridge regression
                ridge_regression = Ridge(alpha=best_alpha)
                ridge_regression.fit(np.array(feat_tr), np.array(y_tr))

                y_hat = torch.from_numpy(ridge_regression.predict(np.array(feat)))
                del ridge_regression

            y_hat_tr = y_hat[idx]
            y_hat_test = y_hat[idx_test]

            tr_cent = y_tr - y_tr.mean(0, keepdim=True)
            test_cent = y_test - y_test.mean(0, keepdim=True)
            gen_cent = y - y.mean(0, keepdim=True)

            # Compute overall (scalar) normalization factors
            tr_norm = (tr_cent**2).mean(0).sum()
            test_norm = (test_cent**2).mean(0).sum()
            gen_norm = (gen_cent**2).mean(0).sum()

            tr_err = ((y_hat_tr - y_tr)**2).mean(0) / tr_norm
            test_err = ((y_hat_test - y_test)**2).mean(0) / test_norm
            gen_err = ((y_hat - y)**2).mean(0) / gen_norm

            r2_tr = 1 - tr_err
            r2_test = 1 - test_err
            r2_gen = 1 - gen_err

            def pearsonr(pred, target):
                yc = target - target.mean(0, keepdim=True)
                yhatc = pred - pred.mean(0, keepdim=True)
                return (yc*yhatc).sum(0)/torch.sqrt((yc**2).sum(0)*(yhatc**2).sum(0))

            pearson_tr = pearsonr(y_hat_tr, y_tr)
            pearson_test = pearsonr(y_hat_test, y_test)
            pearson_gen = pearsonr(y_hat, y)

            errors['gen_norm'][j, i] = gen_norm.cpu().numpy()
            errors['tr_norm'][j, i] = tr_norm.cpu().numpy()
            errors['test_norm'][j, i] = test_norm.cpu().numpy()

            errors['gen_errs'][j, i] = gen_err.cpu().numpy()
            errors['tr_errs'][j, i] = tr_err.cpu().numpy()
            errors['test_errs'][j, i] = test_err.cpu().numpy()

            errors['r2_gen'][j, i] = r2_gen.cpu().numpy()
            errors['r2_tr'][j, i] = r2_tr.cpu().numpy()
            errors['r2_test'][j, i] = r2_test.cpu().numpy()

            errors['pearson_tr'][j, i] = pearson_tr.cpu().numpy()
            errors['pearson_test'][j, i] = pearson_test.cpu().numpy()
            errors['pearson_gen'][j, i] = pearson_gen.cpu().numpy()

    feat, feat_tr, y = 0, 0, 0
    torch.cuda.empty_cache()

    return errors


@torch.no_grad()
def regression_metric(activations, labels, spectrum_dict, cent=True, uncent=False, **kwargs):

    assert type(labels) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"
    assert labels.get('responses') is not None

    reg_responses_uncent = {layer_key: {} for layer_key in activations.keys()}
    reg_responses_cent = {layer_key: {} for layer_key in activations.keys()}
    for layer_key, layer_act in tqdm(activations.items(), total=len(activations), desc='Layer'):
        for label_key, y in labels.items():
            if uncent:
                # Uncentered regression
                eigs = spectrum_dict['uncent'][layer_key]['eigs']
                weights = spectrum_dict['uncent'][layer_key]['weights'][label_key]
                errors = regression(layer_act, y, cent=False, **kwargs)
                reg = errors['reg']
                pvals = errors['pvals']
                theory = gen_error_theory(eigs, weights, reg, pvals=pvals)
                errors |= theory
                reg_responses_uncent[layer_key][label_key] = errors

            if cent:
                # Centered regression
                eigs = spectrum_dict['cent'][layer_key]['eigs']
                weights = spectrum_dict['cent'][layer_key]['weights'][label_key]
                errors = regression(layer_act, y, cent=True, **kwargs)
                reg = errors['reg']
                pvals = errors['pvals']
                theory = gen_error_theory(eigs, weights, reg, pvals=pvals)
                errors |= theory
                reg_responses_cent[layer_key][label_key] = errors

    return {'uncent': reg_responses_uncent,
            'cent': reg_responses_cent}
