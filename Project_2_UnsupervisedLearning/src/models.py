
from itertools import product
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (adjusted_rand_score, silhouette_score,
                            calinski_harabasz_score, davies_bouldin_score)
from sklearn.mixture import GaussianMixture

from .data import balanced_sample, chronological_folds, make_preprocessor
from .config import options

def fit_gmm(x,k, covariance, random_seed, option):
    '''Fit a Gaussian mixture model to the data and return the fitted model.
    The grid search loop in notebooks/03_GMM_TimeOrdered_GridSearch.ipynb calls this function with different settings. 
    The option dictionary contains the n_init and max_iter settings for the GMM.   
    '''
    model = GaussianMixture(n_components=int(k)
                            ,covariance_type=covariance
                            ,reg_covar=1e-5
                            ,n_init=option["n_init"]
                            ,max_iter=option["max_iter"]
                            ,random_state=int(random_seed)
                            )
    # The filter warnings context manager captures convergence warnings 
    # and allows the notebook to continue running without interruption. The warnings are still recorded in the caught list for later inspection if needed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x)
    return model


def cluster_metrics(x, labels, random_seed = 42):
    '''
    Summmarize the sizes and separation of clusters that have already been assigned
    
    Inputs:
    - x: feature matrix, maybe imputed and scaled
    - labels: one predicted cluster label for each row in x
    '''
    # count obs in each cluster
    counts = pd.Series(labels).value_counts()
    result = {
        "n_clusters": len(counts)
        ,"min_share": counts.min() / len(labels) # fraction of obs in smallest cluster
        ,"silhouette": np.nan
        ,"calinski_harabasz": np.nan
        ,"davies_bouldin": np.nan
        }
    
    if 1 < len(counts) < len(labels) and counts.min() > 1: 
        # at least 2 clusters + fewer clusters than obs + at least 2 obs in every cluster
        try:
            result["silhouette"] = silhouette_score(x, labels, sample_size=min(600, len(x)), random_state=random_seed)
        except ValueError:
            # ett sällsynt kluster kan saknas i silhouette-stickprovet
            pass
        result.update(calinski_harabasz=calinski_harabasz_score(x, labels),
                    davies_bouldin=davies_bouldin_score(x, labels))
        
    return result

def run_gmm_cv(rows, feature_sets, cfg):
    '''
    Fit and evaluate many GMM configurations across walk-forward folds and feature sets.
    Return a table of results; selecting the winning configuration happens afterward.
    
    Inputs:
    - rows: the complete development dataset through 2020
    - feature_sets: a dictionary mapping names such as A-D to feature-column lists.
    - cfg: experiment settings, including the computational profile and random seed
    '''
    opt, results = options(cfg=cfg), []
    #loop through the chronological folds (walk-forward folds)
    for fold, train, valid in chronological_folds(development_rows=rows, cfg=cfg):
        train_sample = balanced_sample(frame=train, cap=opt['train_cap'], random_seed=cfg.random_seed)
        valid_sample = balanced_sample(frame=valid, cap=opt['valid_cap'], random_seed=cfg.random_seed + 1)
        # for each feature set:
        for name, columns in feature_sets.items():
            # learn imputation values and scaling statistics from training rows
            prep = make_preprocessor()
            train_prep = prep.fit_transform(train_sample[columns]) # endast foldens träningsperiod för scaling 
            valid_prep = prep.transform(X=valid_sample[columns])
            
            # try every parameter combination 
            # product() generates every combination of component count, covariance type and random seed
            # eg: 3 folds x 4 feature sets x 5 component counts x 2 covariance types x 5 seeds = 600 model fits
            for k, covariance, random_seed in product(opt["k"], opt["covariance"], opt["random_seeds"]):
                record = dict(feature_set=name
                            ,fold=fold
                            ,k=k
                            ,covariance=covariance
                            ,seed=random_seed
                            ,n_train=len(train_sample)
                            ,n_valid=len(valid_sample)
                            ,n_features=len(columns)
                            ,train_end=str(train_sample.Date.max().date())
                            ,valid_start=str(valid_sample.Date.min().date()))
                try:
                    model = fit_gmm(x=train_prep, k=k, covariance=covariance, random_seed= random_seed, option=opt)
                    record.update(valid_log_likelihood=model.score(X=valid_prep)
                                ,train_bic = model.bic(X=train_prep)
                                ,coverged= bool(model.converged_)
                                ,error="")
                except (ValueError, np.linalg.LinAlgError) as error:
                    record.update(valid_log_likelihood=np.nan
                                ,train_bic=np.nan
                                ,converged=False
                                ,error=str(error))
                results.append(record)
                
    return pd.DataFrame(results)
        

def summarize_cv(cv_results):
    summary = cv_results.groupby(["feature_set", "k", "covariance"], as_index=False).agg(
        mean_valid_ll=("valid_log_likelihood", "mean")
        ,std_valid_ll=("valid_log_likelihood", "std")
        ,all_converged=("converged", "all")
        ,n_runs=("converged", "size")
        ,n_scored=("valid_log_likelihood", "count"))
    return summary.sort_values(["feature_set", "mean_valid_ll"], ascending=[True, False])

def fit_candidates(rows, feature_sets, cv_results, cfg):
    opt = options(cfg)
    sample = balanced_sample(rows, opt['train_cap'], cfg.random_seed)
    summary = summarize_cv(cv_results)
    
    artifacts, records, ids = {}, [], {}
    for name, columns in feature_sets.items():
        # keep configuration where all converged and all have a validation score
        eligible = summary.loc[(summary.feature_set == name) & summary.all_converged
                            & summary.n_runs.eq(summary.n_scored)]
        if eligible.empty:
            raise ValueError(f"No converged candiate for {name}. Review gmm_cv.csv; or change experiment")
        best = eligible.iloc[0] # least number of features + highest log-likelihood
        prep = make_preprocessor()
        train_prep = prep.fit_transform(sample[columns])
        model = fit_gmm(train_prep, best.k, best.covariance, cfg.random_seed, opt)
        if not model.converged_:
            raise ValueError(f"The final model fit for feature set {name} did not converge.")
        cid = f"{name}_k{int(best.k)}_{best.covariance}"
        artifacts[cid] = dict(id=cid
                            ,feature_set=name
                            ,features=columns
                            ,preprocessor=prep
                            ,model=model
                            ,train_end=str(rows.Date.max().date())
                            ,sample_keys=sample.index.tolist())
        ids[name] = cid
        records.append(dict(candidate_id=cid
                            ,feature_set=name
                            ,n_features=len(columns)
                            ,k=int(best.k)
                            ,covariance=best.covariance
                            ,cv_ll_within_set=best.mean_valid_ll
                            ,n_train=len(sample)
                            ,**cluster_metrics(train_prep, model.predict(train_prep))))
    return artifacts, pd.DataFrame(records), ids