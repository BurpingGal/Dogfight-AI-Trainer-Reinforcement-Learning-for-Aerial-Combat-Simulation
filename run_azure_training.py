from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment  # <- Added Environment

# Connect to workspace
ws = Workspace.from_config()

# Create experiment
exp = Experiment(workspace=ws, name="Dogfight-PPO-Training")

# Use built-in CPU environment
env = Environment.get(ws, name="AzureML-ACPT-pytorch-1.13-py38-cuda11.7-gpu", version="10")


# Configure run
src = ScriptRunConfig(
    source_directory=".",
    script="train_ppo.py",
    environment=env,
    compute_target="DogfightAITrainer"
)

# Submit
run = exp.submit(src)
print(f"Run submitted: {run.id}")
run.wait_for_completion(show_output=True)
