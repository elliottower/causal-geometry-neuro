"""exp57 variant: structured VAE with z_choice_dim=2."""
from experiments.exp57_structured_vae import run as _run


def run(**kwargs):
    return _run(z_choice_dim=2, **kwargs)
