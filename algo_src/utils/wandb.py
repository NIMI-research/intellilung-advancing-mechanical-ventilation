import wandb


def wandb_init(config: dict, **additional_configs) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group_id"],
        name=config["name"],
        job_type=config["job_type"],
    )
    for key, val in additional_configs.items():
        wandb.run.config[key] = val
    wandb.run.save()


def wandb_resume(entity, project, run_id):
    wandb.init(entity=entity, project=project, id=run_id, resume="must")


def load_wandb_run(entity, project, run_id):
    api = wandb.Api()
    return api.run(path=f"{entity}/{project}/{run_id}")
