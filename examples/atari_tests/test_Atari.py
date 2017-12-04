import os
from btgym.algorithms import AtariRescale42x42, Launcher, BaseAacPolicy, A3C


cluster_config = dict(
    host='127.0.0.1',
    port=12230,
    num_workers=4,  # Set according CPU's available
    num_ps=1,
    num_envs=4,
    log_dir=os.path.expanduser('~/tmp/test_gym_a3c'),
)

env_config = dict(
    class_ref=AtariRescale42x42,  # Gym env. preprocessed to normalized grayscale 42x42 pix.
    kwargs={'gym_id': 'Breakout-v0'}
)

policy_config = dict(
    class_ref=BaseAacPolicy,
    kwargs={}
)

trainer_config = dict(
    class_ref=A3C,
    kwargs=dict(
        opt_learn_rate=[1e-4, 1e-4], # Random log-uniform
        opt_end_learn_rate=1e-5,
        opt_decay_steps=100*10**6,
        model_gae_lambda=0.95,
        model_beta=[0.02, 0.001], # Entropy reg, random log-uniform
        rollout_length=20,
        time_flat=False,
        model_summary_freq=100,
        episode_summary_freq=10,
        env_render_freq=100,
    )
)

launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=True,
    max_env_steps=100*10**6,
    root_random_seed=0,
    verbose=1
)


launcher.run()