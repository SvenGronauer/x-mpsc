"""
    Define default parameters for NPG algorithm.
"""


def defaults():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (400, 300),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (400, 300),
                    'activation': 'relu'},
            # used for model ensemble:
            'hidden_sizes': (50, 50),
            'activation': 'tanh'
        },
        epochs=100,
        gamma=0.99,
        polyak_terminal_set=0.95,
    )


def pendulum():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (64, 64),
                   'activation': 'tanh'},
            'q': {'hidden_sizes': (200, 200),
                    'activation': 'relu'}
        },
        batch_size=1000,
        epochs=100,
        gamma=0.99,
        init_exploration_steps=1000,
    )


def locomotion():
    """Default hyper-parameters for Bullet's locomotion environments.

    Parameters are values suggested in:
    https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/sac.yml
    """
    params = defaults()
    params['mini_batch_size'] = 256
    params['buffer_size'] = 300000
    params['gamma'] = 0.98
    params['epochs'] = 500
    # params['update_after'] = 10000
    # params['update_every'] = 64
    params['lr'] = 1e-3  # default choice is Adam
    return params


# Hack to circumvent kwarg errors with the official PyBullet Envs
def gym_locomotion_envs():
    params = locomotion()
    return params
