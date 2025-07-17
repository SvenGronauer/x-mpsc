"""
    Define default parameters for NPG algorithm.
"""


def defaults():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'},
        },
        batch_size=512,
        epochs=100,
        gamma=0.99,
    )


def cartpole():
    kwargs = defaults()
    kwargs.update(**dict(
        batch_size=256,
        epochs=100,
    ))
    return kwargs


def pendulum():
    kwargs = defaults()
    kwargs.update(**dict(
        batch_size=256,
        epochs=100,
    ))
    return kwargs

