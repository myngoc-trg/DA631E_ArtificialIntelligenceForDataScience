
from itertools import product
from pathlib import Path
import json
import platform
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
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
    The option dictionary controls restarts, iterations, regularization,
    initialization and convergence tolerance. Existing defaults are preserved.
    '''
    model = GaussianMixture(n_components=int(k)
                            ,covariance_type=covariance
                            ,reg_covar=option.get("reg_covar", 1e-5)
                            ,n_init=option["n_init"]
                            ,max_iter=option["max_iter"]
                            ,random_state=int(random_seed)
                            ,tol=option.get("tol", 1e-3)
                            ,init_params=option.get("init_params", "kmeans")
                            )
    # The filter warnings context manager captures convergence warnings 
    # and allows the notebook to continue running without interruption. The warnings are still recorded in the caught list for later inspection if needed.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x)
    model.convergence_warnings_ = [str(item.message) for item in caught]
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

def run_gmm_cv(
    rows, feature_sets, cfg, *, search_options=None
    ,preprocessor_factory=make_preprocessor, checkpoint_path=None, on_result=None
):
    """Run walk-forward CV with shared fitting, optional batching and checkpoints.

    Existing calls retain their defaults. search_options also accepts reg_covar,
    tol, init_params, compute_bic and score_batch_size. on_result receives a copy
    of each completed record so remedy experiments can share this same CV loop.
    """
    check_development_rows(rows, cfg)
    opt = options(cfg=cfg)
    opt.update(search_options or {})
    results = []
    combinations = list(product(opt["k"], opt["covariance"], opt["random_seeds"]))
    first_validation_year = pd.Timestamp(cfg.feature_selection_end).year + 1
    fold_count = pd.Timestamp(cfg.development_end).year - first_validation_year + 1
    total_runs = fold_count * len(feature_sets) * len(combinations)
    started_at = time.perf_counter()
    batch_size = opt.get("score_batch_size")
    if batch_size is not None and (not isinstance(batch_size, int) or batch_size < 1):
        raise ValueError("score_batch_size must be a positive integer or None.")

    print(
        f"Starting GMM CV: {total_runs:,} fits across {fold_count} folds and "
        f"{len(feature_sets)} feature sets.", flush=True
    )
    for fold, train, valid in chronological_folds(rows, cfg):
        print(f"\n{fold}: {len(train):,} training rows, {len(valid):,} validation rows.", flush=True)
        train_sample = balanced_sample(train, opt["train_cap"], cfg.random_seed)
        valid_sample = balanced_sample(valid, opt["validation_cap"], cfg.random_seed + 1)
        print(
            f"  Actual rows used: {len(train_sample):,} training, "
            f"{len(valid_sample):,} validation", flush=True
        )
        for name, columns in feature_sets.items():
            print(f"  Preparing feature set {name} ({len(columns)} features)...", flush=True)
            prep = preprocessor_factory()
            train_prep = prep.fit_transform(train_sample[columns])
            valid_prep = prep.transform(valid_sample[columns]) if batch_size is None else None
            for k, covariance, random_seed in combinations:
                fit_started_at = time.perf_counter()
                record = {
                    "feature_set": name, "fold": fold, "k": k
                    ,"covariance": covariance, "seed": random_seed
                    ,"sampling_seed": cfg.random_seed
                    ,"n_train": len(train_sample), "n_valid": len(valid_sample)
                    ,"n_features": len(columns)
                    ,"train_end": str(train_sample.Date.max().date())
                    ,"valid_start": str(valid_sample.Date.min().date())
                    ,"reg_covar": opt.get("reg_covar", 1e-5)
                    ,"n_init": opt["n_init"], "max_iter": opt["max_iter"]
                    ,"tol": opt.get("tol", 1e-3)
                    ,"init_params": opt.get("init_params", "kmeans")
                }
                try:
                    model = fit_gmm(train_prep, k, covariance, random_seed, opt)
                    if valid_prep is not None:
                        valid_ll = model.score(valid_prep)
                    else:
                        score_sum = 0.0
                        for start in range(0, len(valid_sample), batch_size):
                            batch = valid_sample.iloc[start:start + batch_size]
                            transformed = prep.transform(batch[columns])
                            score_sum += model.score_samples(transformed).sum()
                        valid_ll = score_sum / len(valid_sample)
                    record.update({
                        "valid_log_likelihood": valid_ll
                        ,"train_log_likelihood": model.score(train_prep)
                        ,"train_bic": model.bic(train_prep) if opt.get("compute_bic", True) else np.nan
                        ,"converged": bool(model.converged_), "n_iter": int(model.n_iter_)
                        ,"error": ""
                    })
                except (ValueError, np.linalg.LinAlgError) as error:
                    record.update({
                        "valid_log_likelihood": np.nan, "train_log_likelihood": np.nan
                        ,"train_bic": np.nan, "converged": False, "n_iter": np.nan
                        ,"error": str(error)
                    })
                record["elapsed_seconds"] = time.perf_counter() - fit_started_at
                results.append(record)
                _write_cv_checkpoint(results, checkpoint_path)
                if on_result is not None:
                    on_result(dict(record))
                completed = len(results)
                if completed == 1 or completed % 10 == 0 or completed == total_runs:
                    status = "failed" if record["error"] else "done"
                    print(
                        f"  [{completed:,}/{total_runs:,}] {fold}, set {name}, "
                        f"k={k}, covariance={covariance}, seed={random_seed}: {status} "
                        f"(elapsed {time.perf_counter() - started_at:.1f}s)", flush=True
                    )
    print(
        f"GMM CV complete: {len(results):,}/{total_runs:,} fits in "
        f"{time.perf_counter() - started_at:.1f}s.", flush=True
    )
    return pd.DataFrame(results)
        

def summarize_cv(cv_results):
    summary = cv_results.groupby(["feature_set", "k", "covariance"], as_index=False).agg(
        mean_valid_ll=("valid_log_likelihood", "mean")
        ,std_valid_ll=("valid_log_likelihood", "std")
        ,all_converged=("converged", "all")
        ,n_runs=("converged", "size")
        ,n_scored=("valid_log_likelihood", "count"))
    return summary.sort_values(["feature_set", "mean_valid_ll"], ascending=[True, False])

def shortlist_gmm_configurations(cv_results, top_n=2):
    '''Return the top eligible configurations per feature set without fitting.'''
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    configuration_summary = summarize_cv(cv_results)

    eligible_configurations = configuration_summary.loc[
        configuration_summary["all_converged"]
        & configuration_summary["n_runs"].eq(
            configuration_summary["n_scored"]
        )
    ].copy()

    shortlisted_configurations = (
    eligible_configurations
    .groupby("feature_set", sort=False)
    .head(top_n)
    .reset_index(drop=True)
    )

    return shortlisted_configurations

def fit_candidates(
    rows, feature_sets, cv_results, cfg, *, specifications=None, fit_options=None
    ,model_seed=None, sampling_seed=None, compute_metrics=True
    ,require_convergence=True, preprocessor_factory=make_preprocessor
):
    """Select/refit one candidate per feature set, or refit stated configurations.

    Existing four-argument calls still select each set's best eligible CV row
    and return (artifacts, candidate_table, ids). For explicit configurations,
    pass cv_results=None and specifications=[{"feature_set": ..., "k": ...,
    "covariance": ...}]. fit_options overrides the computational profile.
    Diagnostic repeated fits can skip metrics and report non-convergence rather
    than accepting such a fit as a final candidate.
    """
    check_development_rows(rows, cfg)
    opt = options(cfg)
    opt.update(fit_options or {})
    model_seed = cfg.random_seed if model_seed is None else model_seed
    sampling_seed = cfg.random_seed if sampling_seed is None else sampling_seed
    sample = balanced_sample(rows, opt["train_cap"], sampling_seed)
    summary = summarize_cv(cv_results) if cv_results is not None else None
    if specifications is None:
        if summary is None:
            raise ValueError("Supply CV results or explicit candidate specifications.")
        specifications = []
        for name in feature_sets:
            eligible = summary.loc[
                summary.feature_set.eq(name) & summary.all_converged
                & summary.n_runs.eq(summary.n_scored)
            ]
            if eligible.empty:
                raise ValueError(f"No converged candidate for {name}. Review the CV results.")
            best = eligible.iloc[0]  # Highest validation LL within this feature set.
            specifications.append({
                "feature_set": name, "k": int(best.k), "covariance": best.covariance
            })
    names = [item["feature_set"] for item in specifications]
    if len(set(names)) != len(names):
        raise ValueError("fit_candidates returns one candidate per feature set.")

    artifacts, records, ids = {}, [], {}
    for specification in specifications:
        name = specification["feature_set"]
        columns = feature_sets[name]
        artifact = fit_sampled_candidate(
            sample, columns, specification, cfg, opt, model_seed=model_seed
            ,preprocessor_factory=preprocessor_factory
        )
        if require_convergence and not artifact["converged"]:
            raise ValueError(f"The candidate model fit for feature set {name} did not converge.")
        artifact["sampling_seed"] = sampling_seed
        artifact["training_period_end"] = str(rows.Date.max().date())
        candidate_id = artifact["id"]
        artifacts[candidate_id] = artifact
        ids[name] = candidate_id
        cv_ll = np.nan
        if summary is not None:
            matched = summary.loc[
                summary.feature_set.eq(name)
                & summary.k.eq(artifact["model"].n_components)
                & summary.covariance.eq(artifact["model"].covariance_type)
            ]
            if len(matched) == 1:
                cv_ll = float(matched.iloc[0].mean_valid_ll)
        record = {
            "candidate_id": candidate_id, "feature_set": name
            ,"n_features": len(columns), "k": artifact["model"].n_components
            ,"covariance": artifact["model"].covariance_type
            ,"cv_ll_within_set": cv_ll, "n_train": len(sample)
            ,"converged": artifact["converged"]
        }
        if compute_metrics:
            transformed = artifact["preprocessor"].transform(sample[columns])
            record.update(cluster_metrics(transformed, artifact["model"].predict(transformed)))
        records.append(record)
    return artifacts, pd.DataFrame(records), ids

def predict_states(
    artifact, observations, cfg=None, *, batch_size=20000
    ,include_diagnostics=False
):
    """Assign states/probabilities, optionally adding density and uncertainty.

    Existing two-argument calls return the original columns. Pass cfg for
    development-date guards and include_diagnostics=True for entropy,
    per-observation log-likelihood and raw missing-feature counts. Omitting cfg
    still permits the project's separately authorized, locked final-test path.
    """
    if cfg is not None:
        check_development_rows(observations, cfg)
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    model = artifact["model"]
    features = artifact["features"]
    metadata = [
        name for name in ("Date", "SecuritiesCode", "DayNumber")
        if name in observations
    ]
    results = []
    for start in range(0, len(observations), batch_size):
        batch = observations.iloc[start:start + batch_size]
        transformed = artifact["preprocessor"].transform(batch[features])
        probabilities = model.predict_proba(transformed)
        assignments = batch[metadata].copy()
        assignments["State"] = probabilities.argmax(axis=1)
        assignments["StateProbability"] = probabilities.max(axis=1)
        if include_diagnostics:
            entropy = -np.sum(
                probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)), axis=1
            )
            assignments["normalized_entropy"] = (
                entropy / np.log(model.n_components) if model.n_components > 1 else 0.0
            )
            assignments["log_likelihood"] = model.score_samples(transformed)
            assignments["missing_feature_count"] = batch[features].isna().sum(axis=1)
        results.append(assignments)
    if results:
        return pd.concat(results)
    columns = metadata + ["State", "StateProbability"]
    if include_diagnostics:
        columns += ["normalized_entropy", "log_likelihood", "missing_feature_count"]
    return pd.DataFrame(index=observations.index, columns=columns)

def benchmark_ward_vs_gmm(development_rows, feature_columns, cfg, gmm_covariance_type = "diag"):
    '''Compare Ward hierarchical clustering with a GMM for each candidate number of clusters
    ,using the same sample and feature matrix.
    
    Purpose: check how the 2 unsupervised methods divide the same stock-day observations
    when given the same features and number of clusters.
    
    For each candicate cluster count, it:
    - Samples rows and impute/scales the selected features
    - Fit Ward clustering and a GMM to that same processed data
    - Calculates cluster metrics for each method and saves Ward's labels
    - Returns the metrics table and data needed to examine 
    '''
    settings = options(cfg)
    benchmark_sample = balanced_sample(development_rows, settings['benchmark_cap'], cfg.random_seed +10)
    prep = make_preprocessor()
    X_prep = prep.fit_transform(benchmark_sample[feature_columns])
    metric_records = []
    ward_labels_by_n_clusters = {}
    
    for n_clusters in settings['k']:
        ward_model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric="euclidean")
        ward_labels = ward_model.fit_predict(X_prep)
        ward_labels_by_n_clusters[n_clusters] = ward_labels
        
        metric_records.append({
            'method': "Ward"
            ,'n_clusters': n_clusters
            ,'protocol': "same_training_sample"
            ,'n_samples': len(X_prep)
            ,**cluster_metrics(X_prep,ward_labels)
        })
        
        gmm_model = fit_gmm(X_prep,n_clusters, gmm_covariance_type, cfg.random_seed, settings)
        gmm_labels = gmm_model.predict(X_prep)
        
        metric_records.append({
            'method': "GMM"
            ,'n_clusters': n_clusters
            ,'protocol': "same_training_sample"
            ,'n_samples': len(X_prep)
            ,**cluster_metrics(X_prep,gmm_labels)
        })
    metrics = pd.DataFrame(metric_records)
    benchmark_data = {
        "sample": benchmark_sample
        ,"feature_columns": feature_columns
        ,"scaled_features": X_prep
        ,"ward_labels_by_n_clusters": ward_labels_by_n_clusters
    }
    return metrics, benchmark_data
    
def sample_rows_excluding_random_date_blocks(rows, rng, dates_per_block):
    """Keep rows outside randomly selected blocks of consecutive observed dates.
    
    Because dates are removed as groups, every observation for a given date is retained or removed together. 
    This can be useful for testing whether clustering results are sensitive to which time periods are included.
    """
    sorted_dates = np.sort(rows.Date.unique())
    date_block_ids = np.arange(len(sorted_dates)) // dates_per_block
    unique_block_ids = np.unique(date_block_ids)
    excluded_block_ids = rng.choice(
        unique_block_ids,
        size=max(1, int(np.ceil(0.2 * len(unique_block_ids)))),
        replace=False,
    )
    retained_dates = sorted_dates[~np.isin(date_block_ids, excluded_block_ids)]
    return rows.loc[rows.Date.isin(retained_dates)]


def evaluate_clustering_stability(training_rows, anchor_rows, gmm_artifacts, benchmark_ward_vs_gmm, config):
    """Measure clustering stability: 
    how much the cluster assignments change when the training data has some date blocks removed.
    
    It uses the Adjusted Rand Index (ARI) to compare assignments. ARI is label-order independent, 
    so clusters numbered 0 and 1 can swap numbers without making the result look unstable.    

    For each fitted GMM in artifacts, the function:
        1. Samples an anchor set from anchor and a base training sample from train.
        2. Gets the original model's cluster assignments for the anchor rows.
        3. Repeats the following stability_repeats times:
            - Remove random blocks of dates from the training sample
            - Fit a new preprocessor and GMM on the remaining rows
            - Uses that new model to assign clusters to the same anchor rows
            - Compares the new assignments with the original ones using ARI
        4. Record the ARI, whether the GMM converged and the sample size.
        
    Ward clustering has no built-in predict method for assigning new rows. 
    Instead, the function:
        - removes date blocks from Ward’s original sample, 
        - fits Ward again on the remaining rows, 
        - compares its new assignments with the original Ward assignments for those retained rows only.
    
    Interpretation:
    ARI near 1: the two assignments are very similar.
    ARI near 0: similarity is about what might be expected by chance.
    ARI below 0: agreement is worse than that chance baseline.
    """
    settings = options(config)
    stability_records = []
    anchor_sample = balanced_sample(
        anchor_rows, settings["validation_cap"], config.random_seed + 22
    )
    sampled_training_rows = balanced_sample(
        training_rows, settings["train_cap"], config.random_seed
    )

    for candidate_id, gmm_artifact in gmm_artifacts.items():
        # Keep the evaluation rows fixed so differences reflect refitting, not new observations.
        original_anchor_labels = predict_states(gmm_artifact, anchor_sample)["State"]
        for repeat_index in range(settings["stability_repeats"]):
            repeat_seed = config.random_seed + 100 + repeat_index
            retained_training_rows = sample_rows_excluding_random_date_blocks(
                sampled_training_rows,
                np.random.default_rng(repeat_seed),
                settings["block_dates"],
            )
            repeat_preprocessor = make_preprocessor()
            repeat_training_features = repeat_preprocessor.fit_transform(
                retained_training_rows[gmm_artifact["features"]]
            )
            original_gmm = gmm_artifact["model"]
            refitted_gmm = fit_gmm(
                repeat_training_features
                ,original_gmm.n_components
                ,original_gmm.covariance_type
                ,repeat_seed
                ,settings
            )
            refitted_anchor_labels = refitted_gmm.predict(
                repeat_preprocessor.transform(
                    anchor_sample[gmm_artifact["features"]]
                )
            )
            stability_records.append({
                "method": "GMM"
                ,"candidate_id": candidate_id
                ,"repeat": repeat_index
                ,"n_common": len(anchor_sample)
                ,"ari": adjusted_rand_score(
                    original_anchor_labels, refitted_anchor_labels
                ),
                "converged": bool(refitted_gmm.converged_),
                "protocol": "shared_unseen_2020_anchor",
            })

    # Ward has no predict method, so compare assignments only on retained training rows
    ward_training_sample = benchmark_ward_vs_gmm["sample"]
    ward_feature_columns = benchmark_ward_vs_gmm["feature_columns"]
    for n_clusters, original_ward_labels in benchmark_ward_vs_gmm["ward_labels_by_n_clusters"].items():
        original_labels_by_row = pd.Series(
            original_ward_labels, index=ward_training_sample.index
        )
        for repeat_index in range(settings["stability_repeats"]):
            repeat_seed = config.random_seed + 900 + repeat_index
            retained_ward_rows = sample_rows_excluding_random_date_blocks(
                ward_training_sample,
                np.random.default_rng(repeat_seed),
                settings["block_dates"],
            )
            repeat_preprocessor = make_preprocessor()
            retained_ward_features = repeat_preprocessor.fit_transform(
                retained_ward_rows[ward_feature_columns]
            )
            refitted_ward_labels = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="ward", metric="euclidean"
            ).fit_predict(retained_ward_features)
            stability_records.append({
                "method": "Ward"
                ,"candidate_id": f"Ward_D_k{n_clusters}"
                ,"repeat": repeat_index
                ,"n_common": len(retained_ward_rows)
                ,"ari": adjusted_rand_score(
                    original_labels_by_row.loc[retained_ward_rows.index]
                    ,refitted_ward_labels
                )
                ,"converged": True
                ,"protocol": "common_training_rows"
            })

    return pd.DataFrame(stability_records)


def count_consecutive_trading_day_state_transitions(state_assignments):
    """Count state changes from one market trading da to the next for each stock"""
    sorted_assignments = state_assignments.sort_values(["SecuritiesCode", "Date"]).copy()
    observations_by_stock = sorted_assignments.groupby("SecuritiesCode", sort=False)

    # within each stock, shift Sate and DayNumber to get the previous observation
    previous_state = observations_by_stock["State"].shift(1)
    
    '''
    Keep only rows where the day-number difference is exactly 1
    Exclude the 1st observation per stock and any gaps in is observation.
    
    So the function only counts a transition when it has a previous sate for the same stock 
    and the current row is for the next observed market trading day.
    - 1st obs for a stock has no previous state so it can't form a transition
    - Gap: if a stock has no row for an intervening market day, the function skips
    that pair instead of treating a leter observation as 'next day'
    - Consecutive market day: If DayNumber increases by exactly 1, the previous state -> current state is counted
    '''
    previous_day_number = observations_by_stock["DayNumber"].shift(1)
    is_consecutive_market_day = (sorted_assignments["DayNumber"] - previous_day_number).eq(1)

    # pd.crosstab counts each previous-state -> current-state pair. 
    # Rows are previous states, columns are next states.
    transition_count_table = pd.crosstab(
        previous_state.loc[is_consecutive_market_day],
        sorted_assignments.loc[is_consecutive_market_day, "State"],
    )
    transition_count_table.index.name = "PreviousState"
    transition_count_table.columns.name = "NextState"
    return transition_count_table


def _write_cv_checkpoint(records, checkpoint_path):
    """Persist completed records; this does not implement automatic resume."""
    if checkpoint_path is None:
        return
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(records).to_csv(temporary, index=False)
    temporary.replace(path)


def check_development_rows(rows, cfg):
    """Reject holdout/out-of-period rows and ambiguous row identifiers."""
    if rows.empty or not rows.index.is_unique:
        raise ValueError("Pass non-empty observations with unique row keys.")
    dates = pd.to_datetime(rows["Date"], errors="raise")
    if dates.isna().any() or not dates.between(
        pd.Timestamp(cfg.development_start)
        ,pd.Timestamp(cfg.development_end)
    ).all():
        raise ValueError("Only development-period rows are allowed; exclude 2021.")

def fit_sampled_candidate(
    training_sample, features, specification, cfg, fit_options, model_seed=24
    ,preprocessor_factory=make_preprocessor
):
    """Fit an already sampled training table; fit preprocessing only here."""
    check_development_rows(training_sample, cfg)
    if not features or len(set(features)) != len(features):
        raise ValueError("Supply a non-empty list of distinct feature columns.")
    preprocessor = preprocessor_factory()
    training_matrix = preprocessor.fit_transform(training_sample[features])
    model = fit_gmm(
        training_matrix, specification["k"], specification.get("covariance", "full")
        ,model_seed, fit_options
    )
    feature_set_name = specification["feature_set"]
    candidate_id = (
        f"{feature_set_name}_k{model.n_components}_{model.covariance_type}"
    )
    return {
        "id": candidate_id
        ,"feature_set": feature_set_name
        ,"features": list(features)
        ,"preprocessor": preprocessor
        ,"model": model
        ,"train_end": str(training_sample.Date.max().date())
        ,"train_start": str(training_sample.Date.min().date())
        ,"sample_keys": training_sample.index.tolist()
        ,"n_train": len(training_sample)
        ,"fit_options": dict(fit_options)
        ,"model_seed": int(model_seed)
        ,"converged": bool(model.converged_)
        ,"n_iter": int(model.n_iter_)
        ,"training_log_likelihood": float(model.score(training_matrix))
        ,"warnings": model.convergence_warnings_
    }

def describe_assignments(artifact, rows, assignments, cfg):
    """Return hard cluster sizes, raw/scaled feature medians and uncertainty."""
    check_development_rows(rows, cfg)
    if not rows.index.equals(assignments.index):
        raise ValueError("Rows and assignments must have exactly the same order.")
    features = artifact["features"]
    model = artifact["model"]
    states = assignments["State"]
    counts = states.value_counts().reindex(range(model.n_components), fill_value=0)
    sizes = pd.DataFrame({
        "count": counts
        ,"share": counts / len(rows)
        ,"fitted_mixture_weight": model.weights_
    })
    sizes.index.name = "State"
    scaled = pd.DataFrame(
        artifact["preprocessor"].transform(rows[features])
        ,index=rows.index
        ,columns=features
    )
    uncertainty = assignments.groupby("State").agg(
        count=("State", "size")
        ,mean_max_probability=("StateProbability", "mean")
        ,mean_normalized_entropy=("normalized_entropy", "mean")
        ,share_probability_below_06=("StateProbability", lambda values: values.lt(0.6).mean())
    )
    return {
        "sizes": sizes
        ,"raw_feature_medians": rows[features].groupby(states).median()
        ,"scaled_feature_medians": scaled.groupby(states).median()
        ,"uncertainty": uncertainty
    }

def plot_candidate_diagnostics(
    candidate_id, profiles, *, feature_names=None, output_directory=None
    ,anchor_label="2020 anchor", ambiguous_share_limit=15
):
    """Explain existing assignment summaries visually, without fitting a model.

    The state dashboard covers all six summary columns. A second figure shows
    selected feature medians: cell colors use training-standardized values,
    while annotations use the original feature values with explicit units.
    Training weights are soft mixture weights; anchor shares are hard counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from textwrap import fill

    sizes = profiles["sizes"].copy()
    if sizes.empty or sizes["count"].sum() <= 0:
        raise ValueError("Supply state profiles for a non-empty anchor sample.")
    uncertainty_columns = [
        "mean_max_probability"
        ,"mean_normalized_entropy"
        ,"share_probability_below_06"
    ]
    summary = sizes.join(profiles["uncertainty"][uncertainty_columns])
    observed_ambiguity = summary["share_probability_below_06"].max() * 100
    if not 0 < ambiguous_share_limit <= 100 or observed_ambiguity > ambiguous_share_limit:
        raise ValueError("Increase ambiguous_share_limit to include all observed percentages.")

    positions = np.arange(len(summary))
    anchor_percent = summary["share"].to_numpy() * 100
    training_percent = summary["fitted_mixture_weight"].to_numpy() * 100
    figure_height = max(5.5, 0.58 * len(summary) + 2)
    dashboard, axes = plt.subplots(
        1, 4, figsize=(22, figure_height), sharey=True
        ,gridspec_kw={"width_ratios": [1.45, 1, 1, 1.25]}
    )

    axes[0].barh(
        positions - 0.17, anchor_percent, height=0.28
        ,color="#2476A8", label=f"{anchor_label}: assigned share"
    )
    axes[0].barh(
        positions + 0.17, training_percent, height=0.28
        ,color="#ADB5BD", label="Training: fitted mixture weight"
    )
    common_maximum = max(float(anchor_percent.max()), float(training_percent.max()), 1)
    for position, anchor_value, training_value in zip(positions, anchor_percent, training_percent):
        axes[0].text(
            anchor_value + common_maximum * 0.015, position - 0.17
            ,f"{anchor_value:.2f}%", va="center", fontsize=9, color="#155577"
        )
        axes[0].text(
            training_value + common_maximum * 0.015, position + 0.17
            ,f"{training_value:.2f}%", va="center", fontsize=9, color="#555555"
        )
    axes[0].set_xlim(0, common_maximum * 1.32)
    axes[0].set_title("How common is each state?", fontsize=12, pad=16)
    axes[0].set_xlabel("Share / mixture weight")
    axes[0].xaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].set_yticks(
        positions
        ,labels=[f"State {state}   (n = {int(count):,})" for state, count in summary["count"].items()]
    )
    axes[0].invert_yaxis()

    for axis, column, multiplier, limit, title, label, color in [
        (axes[1], "mean_max_probability", 100, 110
         ,"How strongly is the state preferred?", "Mean winning probability", "#25877C")
        ,(axes[2], "mean_normalized_entropy", 1, 1
          ,"How spread out is membership?", "Mean entropy: 0 = concentrated; 1 = diffuse", "#8B6CAF")
        ,(axes[3], "share_probability_below_06", 100, ambiguous_share_limit
          ,"How often is the assignment ambiguous?", "Share with winning probability below 0.60", "#D48625")
    ]:
        values = summary[column].to_numpy() * multiplier
        axis.barh(positions, values, height=0.48, color=color)
        axis.set_xlim(0, limit)
        axis.set_title(fill(title, 31), fontsize=12, pad=16)
        axis.set_xlabel(fill(label, 35))
        for position, value in zip(positions, values):
            if not np.isfinite(value):
                axis.text(limit * 0.02, position, "No assigned observations", va="center", fontsize=8)
                continue
            text = f"{value:.3f}" if multiplier == 1 else f"{value:.2f}%"
            inside = value + limit * 0.17 > limit
            axis.text(
                value - limit * 0.015 if inside else value + limit * 0.02
                ,position, text, va="center", ha="right" if inside else "left"
                ,fontsize=9, color="white" if inside else "#333333"
            )
        if multiplier == 100:
            axis.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    # Keep probability axes at 0--100%; extra right margin only holds labels.
    axes[1].set_xticks([0, 25, 50, 75, 100])
    for axis in axes:
        axis.grid(axis="x", alpha=0.18)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)


    handles, labels = axes[0].get_legend_handles_labels()
    dashboard.legend(
        handles, labels, loc="lower left", bbox_to_anchor=(0.075, 0.105)
        ,ncol=2, frameon=False, fontsize=10
    )
    dashboard.tight_layout(rect=(0, 0.20, 1, 0.95))

    raw_medians = profiles["raw_feature_medians"]
    scaled_medians = profiles["scaled_feature_medians"]
    features = list(raw_medians.columns) if feature_names is None else list(feature_names)
    if not features or any(name not in raw_medians.columns for name in features):
        raise ValueError("Choose profile features present in this candidate.")
    scaled_values = scaled_medians.reindex(summary.index)[features]
    raw_values = raw_medians.reindex(summary.index)[features]
    feature_labels = {
        "LogReturn1Day": "Daily log return\n(100 x log return)"
        ,"Momentum5Days": "5-day\nmomentum (%)"
        ,"Momentum20Days": "20-day\nmomentum (%)"
        ,"Volatility20Days": "20-day\nvolatility\n(daily, %)"
        ,"Volatility60Days": "60-day\nvolatility\n(daily, %)"
        ,"VolatilityRatio20To60": "Recent / longer\nvolatility ratio"
        ,"CloseToMovingAverage20Days": "Price versus\n20-day average (%)"
        ,"VolumeVs20Days": "Relative log\nvolume deviation"
    }
    multiply_by_100 = {
        "LogReturn1Day", "Momentum5Days", "Momentum20Days"
        ,"Volatility20Days", "Volatility60Days", "CloseToMovingAverage20Days"
    }
    profile_figure, profile_axis = plt.subplots(
        figsize=(max(12, 1.65 * len(features)), max(5, 0.52 * len(summary) + 2.5))
    )
    image = profile_axis.imshow(
        np.ma.masked_invalid(scaled_values.to_numpy()), aspect="auto"
        ,cmap="RdBu_r", vmin=-3, vmax=3
    )
    for row_position in range(len(summary)):
        for column_position, feature in enumerate(features):
            value = raw_values.iloc[row_position, column_position]
            scaled_value = scaled_values.iloc[row_position, column_position]
            if pd.isna(value):
                text = "missing"
            elif feature in multiply_by_100:
                displayed_value = value * 100
                displayed_value = 0.0 if abs(displayed_value) < 0.005 else displayed_value
                text = f"{displayed_value:.2f}"
            elif feature in feature_labels:
                text = f"{value:.2f}"
            else:
                text = f"{value:.3g}"
            profile_axis.text(
                column_position, row_position, text, ha="center", va="center", fontsize=10
                ,color="white" if pd.notna(scaled_value) and abs(scaled_value) > 1.8 else "#222222"
            )
    profile_axis.set_xticks(
        range(len(features))
        ,labels=[feature_labels.get(name, fill(name, 20)) for name in features]
        ,fontsize=9
    )
    profile_axis.set_yticks(
        positions, labels=[f"State {state}" for state in summary.index]
    )
    profile_axis.set_title(
        f"{candidate_id}: what makes each state different?\n"
        
        ,pad=18
    )
    profile_figure.text(
        0.01, 0.025
        ,"Red means above the training mean; blue means below it"
        "Colors are clipped at -3 and +3.\n"
    
        ,fontsize=10, va="bottom"
    )
    profile_figure.tight_layout(rect=(0, 0.11, 1, 1))
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        dashboard.savefig(output_directory / f"{candidate_id}_state_dashboard.png", dpi=150)
        profile_figure.savefig(output_directory / f"{candidate_id}_readable_feature_profiles.png", dpi=150)
        summary.to_csv(output_directory / f"{candidate_id}_state_summary.csv")
    return {
        "state_dashboard": dashboard
        ,"feature_profiles": profile_figure
        ,"state_summary": summary
    }

def assignment_stability(
    training_rows, anchor_rows, feature_sets, specifications, reference_artifacts
    ,cfg, fit_options, model_seeds=(4, 14), sampling_seeds=(142, 242)
):
    """Separate initialization sensitivity from training-sample sensitivity.

All comparisons use the same later anchor observations. ARI handles arbitrary
cluster label permutations. The anchor is development data, not a fresh test.
"""
    check_development_rows(training_rows, cfg)
    check_development_rows(anchor_rows, cfg)
    if training_rows.Date.max() >= anchor_rows.Date.min():
        raise ValueError("Anchor dates must follow every training date.")
    reference_by_set = {
        value["feature_set"]: value for value in reference_artifacts.values()
    }
    records = []
    for specification in specifications:
        name = specification["feature_set"]
        reference = reference_by_set[name]
        if reference["fit_options"] != {**options(cfg), **fit_options}:
            raise ValueError("Use the reference fitting budget for stability checks.")
        if (
            reference["features"] != list(feature_sets[name])
            or reference["model"].n_components != int(specification["k"])
            or reference["model"].covariance_type != specification.get("covariance", "full")
        ):
            raise ValueError("Reference and refitted configurations must match.")
        reference_labels = predict_states(
            reference
            ,anchor_rows
            ,cfg
        )["State"]
        reference_sampling_seed = reference["sampling_seed"]
        designs = [
            ("initialization", reference_sampling_seed, seed)
            for seed in model_seeds if seed != reference["model_seed"]
        ] + [
            ("training_sample", seed, reference["model_seed"])
            for seed in sampling_seeds if seed != reference_sampling_seed
        ]
        for sensitivity, sample_seed, model_seed in designs:
            fitted = fit_candidates(
                training_rows
                ,feature_sets
                ,None
                ,cfg
                ,specifications=[specification]
                ,fit_options=fit_options
                ,model_seed=model_seed
                ,sampling_seed=sample_seed
                ,compute_metrics=False
                ,require_convergence=False
            )[0]
            refitted = next(iter(fitted.values()))
            labels = predict_states(
                refitted
                ,anchor_rows
                ,cfg
            )["State"]
            records.append({
                "candidate_id": reference["id"]
                ,"sensitivity": sensitivity
                ,"sampling_seed": sample_seed
                ,"model_seed": model_seed
                ,"n_anchor": len(anchor_rows)
                ,"converged": refitted["converged"]
                ,"reference_converged": reference["converged"]
                ,"ari_to_reference": adjusted_rand_score(reference_labels, labels)
            })
            print(f"{reference['id']}: {sensitivity}, seed {model_seed}/{sample_seed}", flush=True)
    return pd.DataFrame(records)

def refit_recorded_fold(
    development_rows, feature_sets, cfg, fit_options, feature_set="B", k=5
    ,fold_name="validate_2018", model_seed=24, sampling_seed=None
):
    """Reconstruct one CV fit without using validation observations to fit it."""
    for current_fold, training_rows, validation_rows in chronological_folds(development_rows, cfg):
        if current_fold == fold_name:
            specification = {"feature_set": feature_set, "k": k, "covariance": "full"}
            artifacts = fit_candidates(
                training_rows
                ,feature_sets
                ,None
                ,cfg
                ,specifications=[specification]
                ,fit_options=fit_options
                ,model_seed=model_seed
                ,sampling_seed=sampling_seed
                ,compute_metrics=False
                ,require_convergence=False
            )[0]
            artifact = next(iter(artifacts.values()))
            return artifact, training_rows.loc[artifact["sample_keys"]], validation_rows
    raise ValueError(f"Unknown fold: {fold_name}")

def covariance_diagnostics(artifact):
    """Eigenvalues refer to the fitted, standardized feature space."""
    model = artifact["model"]
    if model.covariance_type != "full":
        raise ValueError("This diagnostic expects full covariance.")
    eigenvalues = np.linalg.eigvalsh(model.covariances_)
    return pd.DataFrame({
        "State": np.arange(model.n_components)
        ,"weight": model.weights_
        ,"minimum_eigenvalue": eigenvalues.min(axis=1)
        ,"maximum_eigenvalue": eigenvalues.max(axis=1)
        ,"condition_number": eigenvalues.max(axis=1) / eigenvalues.min(axis=1)
    })

def likelihood_tail_diagnostics(
    artifact, validation_rows, assignments, tail_fraction=0.001, worst_count=30
):
    """Describe worst scores; do not remove them from reported validation LL."""
    if not 0 < tail_fraction < 1 or not validation_rows.index.equals(assignments.index):
        raise ValueError("Use aligned observations and 0 < tail_fraction < 1.")
    audit = assignments.copy()
    tail_count = max(1, int(np.ceil(len(audit) * tail_fraction)))
    worst_keys = audit["log_likelihood"].nsmallest(tail_count).index
    audit["in_worst_tail"] = audit.index.isin(worst_keys)
    financial_features = [
        name for name in artifact["features"]
        if name.startswith("Fin_") or name == "FinancialAgeDays"
    ]
    audit["any_financial_missing"] = validation_rows[financial_features].isna().any(axis=1)
    event_features = [
        name for name in ("HasExpectedDividend", "SupervisionFlag", "AdjustmentEventFlag")
        if name in validation_rows
    ]
    audit[event_features] = validation_rows[event_features]
    scores = audit["log_likelihood"]
    quantiles = scores.quantile([0, 0.0001, 0.001, 0.01, 0.5, 0.99, 1]).to_frame("log_likelihood")
    tail_summary = audit.groupby("in_worst_tail").agg(
        count=("log_likelihood", "size")
        ,mean_ll=("log_likelihood", "mean")
        ,sum_ll=("log_likelihood", "sum")
    )
    tail_summary["contribution_to_full_mean"] = tail_summary["sum_ll"] / len(audit)
    group_reports = {}
    for column in ["Date", "SecuritiesCode", "State", "any_financial_missing"] + event_features:
        if column not in audit:
            continue
        report = audit.groupby(column, dropna=False).agg(
            count=("log_likelihood", "size")
            ,tail_count=("in_worst_tail", "sum")
            ,mean_ll=("log_likelihood", "mean")
            ,sum_ll=("log_likelihood", "sum")
        )
        report["tail_rate"] = report["tail_count"] / report["count"]
        report["tail_enrichment"] = report["tail_rate"] / audit["in_worst_tail"].mean()
        report["contribution_to_full_mean"] = report["sum_ll"] / len(audit)
        group_reports[column] = report.sort_values("contribution_to_full_mean")
    selected = scores.nsmallest(worst_count).index
    extra_features = [name for name in artifact["features"] if name not in audit]
    worst_rows = audit.loc[selected].join(validation_rows.loc[selected, extra_features])
    return {
        "audit": audit, "quantiles": quantiles, "tail_summary": tail_summary
        ,"group_reports": group_reports, "worst_rows": worst_rows
    }

def feature_shift_report(artifact, training_sample, validation_rows):
    """Compare raw missingness/ranges and transformed validation extremes."""
    features = artifact["features"]
    transformed = artifact["preprocessor"].transform(validation_rows[features])
    absolute_transformed = np.abs(transformed)
    return pd.DataFrame({
        "training_missing_share": training_sample[features].isna().mean()
        ,"validation_missing_share": validation_rows[features].isna().mean()
        ,"training_minimum": training_sample[features].min()
        ,"training_maximum": training_sample[features].max()
        ,"validation_minimum": validation_rows[features].min()
        ,"validation_maximum": validation_rows[features].max()
        ,"validation_abs_scaled_q99": np.quantile(absolute_transformed, 0.99, axis=0)
        ,"validation_abs_scaled_max": absolute_transformed.max(axis=0)
        ,"validation_share_abs_scaled_over_10": (absolute_transformed > 10).mean(axis=0)
    }).sort_values("validation_abs_scaled_max", ascending=False)

def compare_validation_coverage(validation_rows, assignments, sampling_seed=43, cap=50000):
    """Compare cached scores from one model; never refit for these subsets."""
    if not validation_rows.index.equals(assignments.index):
        raise ValueError("Scores must align exactly with validation observations.")
    sample = balanced_sample(validation_rows, cap, sampling_seed)
    in_sample = assignments.index.isin(sample.index)
    records = []
    for name, mask in [
        ("complete_year", np.ones(len(assignments), dtype=bool))
        ,("fixed_sample", in_sample)
        ,("outside_sample", ~in_sample)
    ]:
        values = assignments.loc[mask, "log_likelihood"]
        records.append({
            "validation_subset": name, "count": len(values)
            ,"mean_ll": values.mean(), "minimum_ll": values.min()
            ,"q001_ll": values.quantile(0.001) if len(values) else np.nan
        })
    return pd.DataFrame(records), sample.index

def run_remedy_cv(
    development_rows, base_features, cfg, fit_options, variants
    ,model_seeds=(4, 14, 24), k=5, checkpoint_path=None
):
    """Evaluate prechosen remedies through the existing run_gmm_cv engine.

    Each variant can override fit options, features and preprocessor_factory.
    Keep row budgets matched; raw LL is not comparable across feature spaces.
    """
    records = []
    for variant_name, variant in variants.items():
        current_options = {**fit_options, **variant.get("fit_options", {})}
        if current_options["train_cap"] != fit_options["train_cap"]:
            raise ValueError("Keep training caps equal to isolate the remedy.")
        features = variant.get("features", base_features)
        factory = variant.get("preprocessor_factory", make_preprocessor)
        search_options = {
            **current_options
            ,"k": [k], "covariance": ["full"], "random_seeds": list(model_seeds)
            ,"validation_cap": None, "score_batch_size": 20000, "compute_bic": False
        }

        def collect_result(result):
            result.update({
                "variant": variant_name
                ,"features": json.dumps(features)
                ,"fit_options": json.dumps(current_options, sort_keys=True)
                ,"preprocessor_factory": getattr(factory, "__name__", str(factory))
                ,"train_ll": result["train_log_likelihood"]
                ,"valid_ll": result["valid_log_likelihood"]
            })
            records.append(result)
            _write_cv_checkpoint(records, checkpoint_path)

        run_gmm_cv(
            development_rows, {"B": features}, cfg, search_options=search_options
            ,preprocessor_factory=factory, on_result=collect_result
        )
    return pd.DataFrame(records)

def save_candidate_bundle(artifacts, candidate_table, output_directory):
    """Save updated candidates to a separate directory; do not select a winner."""
    output_directory = Path(output_directory)
    (output_directory / "models").mkdir(parents=True, exist_ok=True)
    if any(not value["converged"] for value in artifacts.values()):
        raise ValueError("Do not save non-converged fits as accepted candidates.")
    ids = {}
    manifest = {}
    for candidate_id, artifact in artifacts.items():
        if artifact["feature_set"] in ids:
            raise ValueError("Save only one final candidate per feature set in a bundle.")
        joblib.dump(artifact, output_directory / "models" / f"{candidate_id}.joblib")
        ids[artifact["feature_set"]] = candidate_id
        manifest[candidate_id] = {
            key: artifact[key] for key in (
                "features", "fit_options", "sampling_seed", "model_seed"
                ,"train_start", "train_end", "n_train", "converged", "n_iter"
            )
        }
    candidate_table.to_csv(output_directory / "gmm_candidates.csv", index=False)
    for filename, value in [("candidate_ids.json", ids), ("fit_manifest.json", manifest)]:
        (output_directory / filename).write_text(
            json.dumps(value, indent=2, default=str), encoding="utf-8"
        )
    (output_directory / "library_versions.json").write_text(
        json.dumps({
            "python": platform.python_version()
            ,"numpy": np.__version__, "pandas": pd.__version__
            ,"scikit_learn": sklearn.__version__
        }, indent=2), encoding="utf-8"
    )
    return ids
