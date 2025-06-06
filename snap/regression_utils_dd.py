import numpy as np
from scipy import optimize
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from snap.ridge_gcv_mod import RidgeCVMod

import pickle

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


def gen_error_theory(eigs, weights, reg, pvals=None, empirical_only=False):

    if empirical_only:
        errors = {'pvals_theory': pvals,
                'kappa': None,
                'gamma': None,
                'eff_regs': None,
                'E_i': None,
                'gen_theory': None,
                'tr_theory': None,
                'radius_theory': None,
                'dimension_theory': None,
                'error_modes_theory': None,
                }

    else:
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
        print(f'E_i sum: {errors["E_i"].sum(-1).sum(-1)}')

    return errors


@torch.no_grad()
def regression(feat, y, pvals=None, cent=False, 
               num_trials=3, reg=None, alpha_per_target=False, 
               scoring='explained_variance', with_pca=True, 
               scale_feats=True, scale_y=True,
               n_folds=5, random_state=0, layer=None,
               name=None, pretrained=None, **kwargs):

    P, N = feat.shape
    C = y.shape[-1]

    if pvals is None:
        pvals = [int(.6*P), int(.8*P)]
    elif isinstance(pvals, (int, float)):
        pvals = [pvals]

    if cent:
        with_mean = True
    else:
        with_mean = False

    if alpha_per_target:
        err_reg = np.zeros((len(pvals), y.shape[1]))
    else:
        err_reg = np.zeros(len(pvals))

    errors = {'pvals': pvals,
              'P': P,
              'N': N,
              'C': C,
              'cent': cent,
              'reg': err_reg, 

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

            idx, idx_test = train_test_split(np.arange(0, P, 1), train_size=p, random_state=j)
            assert len(set(idx)) == p
            assert len(set(idx_test)) == P - p

            y_tr = y[idx]
            y_test = y[idx_test]
            feat_tr = feat[idx]

            feat_scaler = StandardScaler(with_mean=(with_mean and scale_feats), 
                                         with_std=scale_feats)
            if best_alpha is None:
                ridge_reg = Ridge()
            else:
                ridge_reg = Ridge(alpha=best_alpha)

            if with_pca:
                pca_file_name = f'model_{name}_pretrained_{pretrained}_layer_{layer}_p_{p}_cent_{cent}_scaled_{scale_feats}_trial_{j}'
                path = f'/mnt/home/alargen/SNAP/snap_analysis_data/pca_decomps/{pca_file_name}'
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        all_feat = pickle.load(f)
                        feat_tr = all_feat[idx]
                else:
                    norm_feat_tr = feat_scaler.fit_transform(feat_tr)
                    norm_feat = feat_scaler.transform(feat)

                    pca = PCA(n_components=p)
                    feat_tr = pca.fit_transform(norm_feat_tr)
                    all_feat = pca.transform(norm_feat)
                    with open(path, 'wb') as f:
                        pickle.dump(all_feat, f)

                pipeline = Pipeline([
                    ('ridge', ridge_reg)
                ])
            else:
                pipeline = Pipeline([
                    ('feat_scaler', feat_scaler),
                    ('ridge', ridge_reg)
                ])
                all_feat = feat

            y_scaler = StandardScaler(with_mean=(with_mean and scale_y), 
                                      with_std=scale_y)
            regr = TransformedTargetRegressor(regressor=pipeline,
                                              transformer=y_scaler)

            if best_alpha is None: # need to search for good alpha
                param_grid = {'regressor__ridge__alpha': alphas}
                shuffle = not alpha_per_target # don't want folds to be different when fitting to each voxel
                if shuffle:
                    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
                else:
                    kf = KFold(n_splits=n_folds, shuffle=shuffle)
                gs = GridSearchCV(regr, param_grid, cv=kf, scoring=scoring, n_jobs=-1)

                if alpha_per_target: # pipeline doesn't corretly handle RidgeCV (or generally EstimatorCV)
                    best_alpha = torch.zeros(y.shape[1]) # one per voxel
                    y_hat = torch.zeros_like(y) # each col is a voxel

                    for y_idx in range(y.shape[1]):
                        single_y_tr = y_tr[:, y_idx]
                        
                        gs.fit(np.array(feat_tr), np.array(single_y_tr))
                        best_alpha[y_idx] = gs.best_params_['regressor__ridge__alpha']

                        single_y_hat = torch.from_numpy(gs.predict(np.array(all_feat)))
                        y_hat[:, y_idx] = single_y_hat
                    best_alpha = best_alpha.numpy()

                    print_alpha = (min(best_alpha), max(best_alpha))

                else:
                    gs.fit(np.array(feat_tr), np.array(y_tr))
                    best_alpha = gs.best_params_['regressor__ridge__alpha']
                    
                    y_hat = torch.from_numpy(gs.predict(np.array(all_feat)))

                    print_alpha = best_alpha

                print(f'\n N: {N}, p: {p}, Best Alpha: {print_alpha}, with pca: {with_pca}, feat_scaler: {feat_scaler}')
                errors['reg'][i] = best_alpha/p

                del gs, kf, param_grid

            else: #have alpha, do regression as normal
                regr.fit(np.array(feat_tr), np.array(y_tr))

                y_hat = torch.from_numpy(regr.predict(np.array(all_feat)))

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

    del feat, feat_tr, all_feat #, norm_feat, norm_feat_tr, feat_pca, feat_tr_pca
    del y, y_tr, y_test, y_hat, y_hat_tr, y_hat_test # norm_y, norm_y_tr, norm_y_test,
    del regr
    # feat, feat_tr, y = 0, 0, 0
    torch.cuda.empty_cache()

    return errors


@torch.no_grad()
def regression_metric(activations, labels, spectrum_dict, cent=True, uncent=False, empirical_only=False, **kwargs):

    assert type(labels) is dict, "labels should be provided as a dict (e.g. {'classes': classes})"
    assert labels.get('responses') is not None

    reg_responses_uncent = {layer_key: {} for layer_key in activations.keys()}
    reg_responses_cent = {layer_key: {} for layer_key in activations.keys()}
    for layer_key, layer_act in tqdm(activations.items(), total=len(activations), desc='Layer'):
        for label_key, y in labels.items():
            if uncent:
                # Uncentered regression
                if not empirical_only:
                    eigs = spectrum_dict['uncent'][layer_key]['eigs']
                    weights = spectrum_dict['uncent'][layer_key]['weights'][label_key]
                else:
                    eigs = None
                    weights = None
                errors = regression(layer_act, y, cent=False, layer=layer_key, **kwargs)
                reg = errors['reg']
                pvals = errors['pvals']
                theory = gen_error_theory(eigs, weights, reg, pvals=pvals, empirical_only=empirical_only)
                errors |= theory
                reg_responses_uncent[layer_key][label_key] = errors

            if cent:
                # Centered regression
                if not empirical_only:
                    eigs = spectrum_dict['cent'][layer_key]['eigs']
                    weights = spectrum_dict['cent'][layer_key]['weights'][label_key]
                else:
                    eigs = None
                    weights = None
                errors = regression(layer_act, y, cent=True, layer=layer_key, **kwargs)
                reg = errors['reg']
                pvals = errors['pvals']
                theory = gen_error_theory(eigs, weights, reg, pvals=pvals, empirical_only=empirical_only)
                errors |= theory
                reg_responses_cent[layer_key][label_key] = errors

    return {'uncent': reg_responses_uncent,
            'cent': reg_responses_cent}
