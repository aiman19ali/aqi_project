try:
    import hopsworks
except Exception:
    hopsworks = None
import pandas as pd
from typing import Optional


def fetch_from_feature_store(
    project_name: Optional[str],
    feature_group_name: str,
    version: int = 1
) -> pd.DataFrame:
    """Fetch features from Hopsworks Feature Store.

    This function is defensive:
      - If the hopsworks SDK isn't installed it raises a helpful error.
      - It tries a flexible login call to support different client versions.
    """
    if hopsworks is None:
        raise RuntimeError("hopsworks SDK is not installed. Install with: pip install hopsworks")
    try:
        if project_name:
            project = hopsworks.login(project=project_name)
        else:
            project = hopsworks.login()
    except TypeError:

        project = hopsworks.login()
    fs = project.get_feature_store()


    try:
        fg = fs.get_feature_group(name=feature_group_name, version=version)
    except Exception:

        try:
            fg = fs.get_or_create_feature_group(name=feature_group_name, version=version)
        except Exception as e:
            raise RuntimeError(f"Could not access feature group '{feature_group_name}:{version}': {e}")
    return fg.read()