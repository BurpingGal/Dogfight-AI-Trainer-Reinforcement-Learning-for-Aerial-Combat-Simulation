from azureml.core import Workspace, Environment

# Connect to workspace
ws = Workspace.from_config()

# List environments safely
print("Available environments:\n")
envs = Environment.list(workspace=ws)

for env_name, env in envs.items():
    try:
        print(f"Name: {env_name}, Version: {env.version}")
    except Exception as e:
        print(f"Name: {env_name}, Error retrieving details: {e}")
