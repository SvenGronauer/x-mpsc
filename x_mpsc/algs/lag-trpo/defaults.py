"""
    Define default parameters for Lagrangian-TRPO algorithm.
"""


def defaults():
    return dict(
        actor='mlp',
        ac_kwargs={
            'pi': {'hidden_sizes': (64, 64),
                   'activation': 'tanh'},
            'val': {'hidden_sizes': (64, 64),
                    'activation': 'tanh'}
        },
        adv_estimation_method='gae',
        cost_limit=0.,
        epochs=300,
        gamma=0.99,
        steps_per_epoch=8 * 1000,
        target_kl=1e-3,
        use_exploration_noise_anneal=True
    )
