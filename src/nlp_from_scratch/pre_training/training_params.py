from pydantic import BaseModel


class TrainingParams(BaseModel):
    accumulate_grad_batches: int
    log_every_n_steps: int
    batch_size: int
    n_iterations: int
