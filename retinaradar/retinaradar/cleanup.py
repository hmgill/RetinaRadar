
"""

"""

from pathlib import Path
import shutil

from retinaradar.paths import PATHS


def cleanup_files_subdirs(dir_to_cleanup):
    path = Path(dir_to_cleanup)

    for item in path.iterdir():
        # remove dirs
        if item.is_dir():
            shutil.rmtree(item)
        # remove files
        else:
            item.unlink()


def cleanup_all():
    path = Path(PATHS["retinaradar_runs"])
    cleanup_files_subdirs(path)

    
def cleanup_id(run_id):
    path = Path(PATHS["retinaradar_runs"], run_id)
    cleanup_files_subdirs(path)
    shutil.rmtree(path)

    
