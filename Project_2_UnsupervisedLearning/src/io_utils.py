"""Konfiguration, sparade steg och kontrollpunkten före 2021."""
import hashlib
import json
from pathlib import Path
from dataclasses import asdict

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    
def sha256(path):
    '''Compute the SHA256 hash of a file.
    Create a fingerprint to detect changed file contents
    '''
    # create the SHA256 calculator
    digest = hashlib.sha256()
    
    #open file as raw bytes
    with Path(path).open("rb") as file:
        # read the file in 1MB blocks to avoid using too much memory
        # process those bytes and repear until reading returns b"" ie empty bytes indicating the end of file 
        for block in iter(lambda: file.read(1024 * 1024), b""):
            # add each block to the calculation
            digest.update(block)
            
    # Return the fingerprint as hexadecimal text 
    return digest.hexdigest() 


def prepare_run(root, cfg):
    """Create output/run_name and reject reuse with a different configuration."""
    
    #build experiment's output path and create the folder if it does not exist
    out = Path(root) / "output" / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "experiment_config.json"
    
    # check any previously saved configuration
    # asdict(cfg) converts dataclass into a dictionary, which can be compared to the saved configuration
    if config_path.exists() and read_json(config_path)["config"] != asdict(cfg):
        raise RuntimeError("Configuration changed. Choose a new run_name and rerun notebooks 01-02.")
    if not config_path.exists():
        # record the config for a new run
        write_json(config_path, {"config": asdict(cfg),
                                "row_unit": "SecuritiesCode + Date"})
    for name in ["models", "tables", "figures"]:
        (out / name).mkdir(exist_ok=True)
    return out

def require_files(out, names):
    '''Check that files needed by the next step exist. If not, raise an error with a message about which files are missing.'''
    
    # check each filename inside out and collect the missing ones
    missing = [name for name in names if not (Path(out) / name).exists()]
    if missing:
        raise FileNotFoundError("Kör tidigare notebooks först. Saknas: " + ", ".join(missing))




def record_input_checksums(root, cfg, out):
    """Record the experiment's source files, including Project 1's feature audit."""
    paths = []
    for name in ("project1_features_path", "project1_feature_audit_path", "stock_list_path", "pred_path"):
        source = getattr(cfg, name)
        if source is None:
            continue
        source_path = (Path(root) / source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing input file for {name}: {source_path}")
        paths.append(source_path)
    checksums = {str(path): sha256(path) for path in paths}
    checksum_path = Path(out) / "source_checksums.json"
    if checksum_path.exists() and read_json(checksum_path) != checksums:
        raise RuntimeError("Source files changed. Choose a new run_name.")
    write_json(checksum_path, checksums)


def save_figure(fig, out, filename):
    fig.tight_layout()
    fig.savefig(Path(out) / "figures" / filename, dpi=150, bbox_inches="tight")


def finalize_model_selection(root, out, cfg, selected_model_id, reason):
    """Lock the selected pipeline after its final refit through development_end.

    final_model_fit.json records selected_model_id, feature_set, features,
    training_start, training_end and n_rows. The dates describe the configured
    fit period; n_rows must equal the full prepared development dataset.
    """
    if not reason.strip():
        raise ValueError("Write a model-selection justification before the final test.")
    root, out = Path(root), Path(out)
    required = ["feature_catalog.csv", "feature_sets.json", "feature_filter_fit.json", "folds.csv",
                "gmm_cv.csv", "gmm_candidates.csv", "candidate_ids.json", "final_model_fit.json",
                "source_checksums.json", "experiment_config.json", "data_audit.json"]
    require_files(out, required)
    candidates = read_json(out / "candidate_ids.json")
    if selected_model_id not in candidates.values():
        raise ValueError("The selected model must be in the candidate list.")
    model_file = "models/" + selected_model_id + ".joblib"
    require_files(out, [model_file])
    required.append(model_file)
    required += [name for name in ("ward_benchmark.csv", "stability.csv") if (out / name).is_file()]

    dataset_audit = read_json(root / cfg.development_audit_path)
    if dataset_audit != read_json(out / "data_audit.json"):
        raise RuntimeError("The prepared dataset changed after this experiment started.")
    fit_record = read_json(out / "final_model_fit.json")
    expected_fit = {"selected_model_id": selected_model_id,
                    "training_start": cfg.development_start, "training_end": cfg.development_end,
                    "n_rows": dataset_audit["rows_after_filter"]}
    if any(fit_record.get(name) != value for name, value in expected_fit.items()):
        raise ValueError("Refit the selected pipeline on all development rows through development_end.")
    feature_sets = read_json(out / "feature_sets.json")
    if fit_record.get("features") != feature_sets.get(fit_record.get("feature_set")) or not fit_record.get("features"):
        raise ValueError("Final-model features do not match the frozen feature set.")
    if sha256(root / cfg.development_features_path) != dataset_audit["dataset_sha256"]:
        raise RuntimeError("The development dataset changed before model locking.")
    if read_json(out / "experiment_config.json")["config"] != asdict(cfg):
        raise RuntimeError("The experiment configuration changed before model locking.")
    for name, expected in read_json(out / "source_checksums.json").items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise RuntimeError("A source file changed before model locking: " + name)

    record = {
        "config": asdict(cfg), "selected_model_id": selected_model_id,
        "candidate_ids": list(candidates.values()), "reason": reason, "final_fit": fit_record,
        "checksums": {name: sha256(out / name) for name in required},
        "dataset_checksums": {name: sha256(root / name) for name in (
            cfg.development_features_path, cfg.development_audit_path
        )},
        "code_checksums": {str(path.relative_to(root)): sha256(path)
                           for path in sorted((root / "src").glob("*.py"))},
    }
    lock = out / "selection_lock.json"
    if lock.exists() and read_json(lock) != record:
        raise RuntimeError("Another model selection is already locked. Choose a new run_name.")
    write_json(lock, record)
    verify_lock(out, cfg)
    return record


def verify_lock(out, cfg):
    """Verify the selected pipeline, development data, sources and code before 2021."""
    out = Path(out).resolve()
    require_files(out, ["selection_lock.json"])
    record = read_json(out / "selection_lock.json")
    if record["config"] != asdict(cfg):
        raise RuntimeError("Configuration differs from the locked experiment.")
    root = out.parents[1]
    for base, checksums in ((out, record["checksums"]), (root, record["dataset_checksums"]),
                            (root, record["code_checksums"] )):
        for name, expected in checksums.items():
            if not (base / name).is_file() or sha256(base / name) != expected:
                raise RuntimeError("A locked file changed: " + name)
    for name, expected in read_json(out / "source_checksums.json").items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise RuntimeError("A source file changed: " + name)
    return record


def ensure_unlocked(out):
    """Prevent development notebooks from overwriting a finalized experiment."""
    if (Path(out) / "selection_lock.json").exists():
        raise RuntimeError("This experiment is finalized. Choose a new run_name for further development.")
