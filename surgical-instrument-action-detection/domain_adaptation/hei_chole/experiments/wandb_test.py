import wandb

# Initialize a new run
wandb.init(project="test-project")

# Log something simple
wandb.log({"test": 1})

# Close the run
wandb.finish()