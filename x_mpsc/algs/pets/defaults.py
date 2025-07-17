"""
    Define default parameters for MuMo algorithms.
"""

MAX_VARIANCE = 0.1
MIN_VARIANCE = 0.001


def defaults():
    return dict(
        actor='mlp',
        ac_kwargs={
            'hidden_sizes': (50, 50),
            'activation': 'tanh'
        },
        buffer_size=int(1e5),
        cem_num_elites=32,
        cem_pop_size=256,
        cem_trajectory_length=25,
        epochs=100,
        num_mini_batches=32,
    )
