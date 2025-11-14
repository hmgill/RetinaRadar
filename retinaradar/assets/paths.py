from datetime import datetime
from pathlib import Path 
import shortuuid


"""
CONSTANT PATHS
"""

retinaradar_root = Path(__file__).parent.absolute()
retinaradar_output = Path(retinaradar_root, "output")
retinaradar_runs = Path(retinaradar_output, "runs")

constant_paths = {
    "retinaradar_root": retinaradar_root,
    "retinaradar_output": retinaradar_output,
    "retinaradar_runs": retinaradar_runs,
}


"""
DYNAMIC PATHS - All outputs consolidated under single run directory
"""

# Unique run id 
run_uuid = shortuuid.uuid()
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
run_id = f"run_{run_uuid}-{run_timestamp}"

# Main run directory
run_output = Path(retinaradar_runs, run_id)

# Organized subdirectories within the run folder
dynamic_paths = {
    "output": run_output,
    "log": Path(run_output, "log"),
    "mlflow": Path(run_output, "mlflow"),  # All MLflow artifacts here
    "checkpoints": Path(run_output, "checkpoints"),  # Model checkpoints
    "models": Path(run_output, "models"),  # Final trained models
    "artifacts": Path(run_output, "artifacts"),  # Metadata, configs, datasets
    "results": Path(run_output, "results"),  # Analysis results, plots, reports
}

extra_info = {
    "run_id": run_id,
    "run_uuid": run_uuid,
    "run_timestamp": run_timestamp,
    "loguru": Path(run_output, "log", f"{run_id}.log")
}


# Create directories when module is imported
def make_dynamic_paths():
    """Create all dynamic path directories"""
    for path in dynamic_paths.values():
        path.mkdir(parents=True, exist_ok=True)

make_dynamic_paths()


"""
PATHS constant - single source of truth for all paths
"""
PATHS = constant_paths | dynamic_paths | extra_info


def get_run_summary():
    """
    Get a formatted summary of the current run's directory structure
    """
    summary = f"""
    Run Directory Structure
    {'='*60}
    Run ID: {PATHS['run_id']}
    Root: {PATHS['output']}
    
    Subdirectories:
      - logs/         : {PATHS['log']}
      - mlflow/       : {PATHS['mlflow']}
      - checkpoints/  : {PATHS['checkpoints']}
      - models/       : {PATHS['models']}
      - artifacts/    : {PATHS['artifacts']}
      - results/      : {PATHS['results']}
    {'='*60}
    """
    return summary


def get_inference_paths(run_directory: Path):
    """
    Given a run directory, return paths to key inference artifacts
    
    Args:
        run_directory: Path to a specific run directory
        
    Returns:
        dict: Dictionary containing paths to model, metadata, and config
    """
    inference_paths = {
        "inference_package": Path(run_directory, "artifacts", "inference_package.json"),
        "label_metadata": Path(run_directory, "artifacts", "label_metadata.json"),
        "dataset": Path(run_directory, "artifacts", "dataset.dill"),
        "final_checkpoint": Path(run_directory, "checkpoints", "last.ckpt"),
        "mlflow_tracking": Path(run_directory, "mlflow"),
    }
    
    return inference_paths
