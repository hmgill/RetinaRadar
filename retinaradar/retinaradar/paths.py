from datetime import datetime
from pathlib import Path 
import shortuuid


"""
CONSTANT PATHS
"""

retinaradar_root = Path(__file__).parent.absolute()
retinaradar_output = Path(retinaradar_root, "output")
retinaradar_runs = Path(retinaradar_output, "runs")
retinaradar_model_output = Path(retinaradar_output, "model_output")
retinaradar_tl_output = Path(retinaradar_model_output, "tl")
retinaradar_agent_output = Path(retinaradar_model_output, "agent")

constant_paths = {
    "retinaradar_output" : retinaradar_output,
    "retinaradar_runs" : retinaradar_runs,
    "retinaradar_tl_output" : retinaradar_tl_output,
    "retinaradar_agent_output" : retinaradar_agent_output
}



"""
DYNAMIC PATHS
"""

# unique run id 
run_uuid = shortuuid.uuid()
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
run_id = f"run_{run_uuid}-{run_timestamp}"

run_output = Path(retinaradar_runs, run_id)
dynamic_paths = {
    "output" : run_output,
    "results" : Path(run_output, "results"),
    "log" : Path(run_output, "log"),
}
loguru_path = {
    "loguru" : Path(run_output, "log", f"{run_id}.log")
}


# Create directories when module is imported
def make_dynamic_paths():
    for path in dynamic_paths.values():
        path.mkdir(parents=True, exist_ok=True)

make_dynamic_paths()



"""
PATHS constant 
"""
PATHS = constant_paths | dynamic_paths | loguru_path 
