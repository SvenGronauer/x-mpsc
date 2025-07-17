"""
    Define default parameters for NPG algorithm.
"""


def defaults():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'}
        },
        epochs=100,
        gamma=0.99,
    )


def pendulum():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'},
        },
        batch_size=256,
        ensemble_hiddens=(10, 10),
        epochs=200,
        init_exploration_steps=8000,
        lr=3e-4,
        model_retain_epochs=10,
        model_train_freq=128,
        mpsc_horizon=7,
        num_model_rollouts=400,
        policy_train_iterations=20,
        pretrain_policy=False,
        rollout_max_length=5,
        rollout_min_length=1,
        rollout_max_epoch=40,
        rollout_min_epoch=10,
        update_every=64,
    )


def cartpole():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'}
        },
        batch_size=256,
        epochs=50,
        ensemble_hiddens=(10, 10),
        init_exploration_steps=8000,
        lr=1e-3,
        model_retain_epochs=20,
        model_train_freq=128,
        num_model_rollouts=400,
        policy_train_iterations=5,
        pretrain_policy=True,
        rollout_max_length=5,
        rollout_min_length=1,
        rollout_max_epoch=20,
        rollout_min_epoch=1,
        update_every=64,
    )


def twolinkarm():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'}
        },
        batch_size=512,
        ensemble_hiddens=(20, 20),
        delay_factor=20,
        epochs=100,
        init_exploration_steps=8000,
        mpsc_feedback_factor=1e-4,
        mpsc_horizon=5,
        policy_train_iterations=20,
        pretrain_policy=False,
        rollout_max_length=3,
        rollout_min_length=1,
        rollout_max_epoch=50,
        rollout_min_epoch=10,
        update_every=64,
        warm_up_ensemble_train_epochs=250,
    )


def drone():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'}
        },
        batch_size=512,
        epochs=50,
        ensemble_hiddens=(20, 20),
        init_exploration_steps=8000,
        lr=1e-3,
        model_retain_epochs=20,
        model_train_freq=128,
        num_model_rollouts=400,
        mpsc_feedback_factor=1e-4,
        mpsc_horizon=4,
        policy_train_iterations=20,
        pretrain_policy=True,
        rollout_max_length=3,
        rollout_min_length=1,
        rollout_max_epoch=20,
        rollout_min_epoch=1,
        update_every=64,
        warm_up_ensemble_train_epochs=250,
    )


def mujoco():
    return dict(
        ac_kwargs={
            'pi': {'hidden_sizes': (100, 100),
                   'activation': 'relu'},
            'q': {'hidden_sizes': (100, 100),
                    'activation': 'relu'}
        },
        batch_size=1024,
        ensemble_hiddens=(50, 50),
        epochs=100,
        lr=1e-3,
        model_retain_epochs=20,
        mpsc_horizon=3,
        init_exploration_steps=8*2048,
        num_model_rollouts=400,
        policy_train_iterations=20,
        rollout_max_length=15,
        rollout_min_length=1,
        rollout_max_epoch=100,
        rollout_min_epoch=20,
        update_every=64,
    )