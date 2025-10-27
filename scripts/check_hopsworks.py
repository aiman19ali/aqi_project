"""
Dry-run Hopsworks connectivity checker.

Run:
  python scripts/check_hopsworks.py

It will:
 - try to import the hopsworks SDK
 - read HOPSWORKS_API_KEY / hopswork_api and HOPSWORKS_PROJECT / hopsworks_project and HOPSWORKS_HOST
 - attempt to login and access the feature store
 - list available feature groups (best-effort)

This script is safe: it only queries metadata and prints helpful diagnostics.
"""

import os
import sys

try:
    import hopsworks
except Exception as e:
    print(f"ERROR: could not import hopsworks: {e}")
    print("Install with: pip install hopsworks")
    sys.exit(1)


def main():
    api_key = os.environ.get("HOPSWORKS_API_KEY") or os.environ.get("hopswork_api")
    project_name = os.environ.get("HOPSWORKS_PROJECT") or os.environ.get("hopsworks_project")
    host = os.environ.get("HOPSWORKS_HOST") or os.environ.get("hopsworks_host")

    print("Hopsworks connectivity checker")
    print(f"Using env: HOPSWORKS_API_KEY={'present' if api_key else 'missing'}, HOPSWORKS_PROJECT={project_name or 'missing'}, HOPSWORKS_HOST={host or 'missing'}")

    # Try login using flexible params
    try:
        login_kwargs = {}
        if api_key:
            login_kwargs["api_key_value"] = api_key
        if host:
            login_kwargs["host"] = host
        if project_name:
            login_kwargs["project"] = project_name

        try:
            if login_kwargs:
                project = hopsworks.login(**login_kwargs)
            else:
                project = hopsworks.login()
        except TypeError:
            # Some client versions expect different arg names or none
            project = hopsworks.login()

    except Exception as e:
        print(f"ERROR: failed to login to Hopsworks: {e}")
        print("Check your HOPSWORKS_API_KEY / hopswork_api and HOPSWORKS_HOST / HOPSWORKS_PROJECT environment variables.")
        sys.exit(2)

    try:
        fs = project.get_feature_store()
    except Exception as e:
        print(f"ERROR: login succeeded but could not get feature store: {e}")
        sys.exit(3)

    print("Connected to feature store. Attempting to list feature groups (best-effort)...")
    try:
        # Different clients expose different listing APIs; try a couple of known methods
        try:
            fgs = fs.get_feature_groups()
            # If this returns a list-like of feature group objects
            print(f"Found {len(fgs)} feature groups (listing up to 50):")
            for fg in fgs[:50]:
                try:
                    name = getattr(fg, 'name', None) or getattr(fg, 'get_name', lambda: None)()
                    version = getattr(fg, 'version', None) or getattr(fg, 'get_version', lambda: None)()
                    print(f" - {name}:{version}")
                except Exception:
                    print(" - (could not read feature group metadata)")
        except Exception:
            # Fallback: try feature store's get_feature_group_names or other metadata
            try:
                names = fs.get_feature_group_names()
                print(f"Found {len(names)} feature groups: {names[:50]}")
            except Exception as e:
                print(f"Could not list feature groups via known APIs: {e}")

    except Exception as e:
        print(f"ERROR while listing feature groups: {e}")
        sys.exit(4)

    print("Hopsworks connectivity check completed successfully.")


if __name__ == '__main__':
    main()
