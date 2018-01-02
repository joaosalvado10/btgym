import os
import backtrader as bt

from btgym import BTgymEnv, BTgymDataset,ExtraLinesDataset
from btgym.strategy.observers import Reward, Position, NormPnL
from btgym.algorithms import Launcher, PPO, Unreal, A3C

from btgym.research import DevStrat_4_6

from btgym.algorithms.policy import Aac1dPolicy
from gym import spaces


# Set backtesting engine parameters:

MyCerebro = bt.Cerebro()




MyCerebro.addstrategy(
    DevStrat_4_6,
    drawdown_call=80, # max % to loose, in percent of initial cash
    target_call=100,  # max % to win, same
    skip_frame=10,
)
# Set leveraged account:
MyCerebro.broker.setcash(2000)
MyCerebro.broker.setcommission(commission=0.0001, leverage=10.0) # commisssion to imitate spread
MyCerebro.addsizer(bt.sizers.SizerFix, stake=2000,)

#MyCerebro.addanalyzer(bt.analyzers.DrawDown)

# Visualisations for reward, position and PnL dynamics:
MyCerebro.addobserver(Reward)
MyCerebro.addobserver(Position)
MyCerebro.addobserver(NormPnL)

MyDataset = ExtraLinesDataset(
    #filename='.data/DAT_ASCII_EURUSD_M1_2010.csv',
    #filename='./data/DAT_ASCII_EURUSD_M1_201704.csv', #USE TO TEST
    filename='./data/Train_EurUsd.csv', #USE TO TRAIN
    start_weekdays={0, 1, 2, 3},
    episode_duration={'days': 120, 'hours': 0, 'minutes': 0},
    start_00=False,
    #time_gap={'hours': 6},
    time_gap={'days': 150}
)

env_config = dict(
    class_ref=BTgymEnv,
    kwargs=dict(
        dataset=MyDataset,
        engine=MyCerebro,
        render_modes=['episode', 'human','external'],
        render_state_as_image=True,
        render_ylabel='OHL_diff.',
        render_size_episode=(12,8),
        render_size_human=(9, 4),
        render_size_state=(11, 4),
        render_dpi=75,
        port=6430,
        data_port=6739,
        connect_timeout=60,
        verbose=0,  # better be 0
    )
)

cluster_config = dict(
    host='127.0.0.1',
    port=12446,
    num_workers= 2,  # Set according CPU's available
    num_ps=1,
    num_envs=1,  # do not change yet
    log_dir=os.path.expanduser('~/tmp/New_features_Unreal-yiel10'),
)

policy_config = dict(
    class_ref=Aac1dPolicy,
    kwargs={}
)

trainer_config = dict(
    class_ref=Unreal,
    kwargs=dict(
        opt_learn_rate=1e-4, # or random log-uniform range, values > 2e-4 can ruin training
        opt_end_learn_rate=1e-5,
        opt_decay_steps=100*10**6,
        model_gamma=0.95,
        model_gae_lambda=1.0,
        model_beta=[0.05, 0.01], # Entropy reg, random log-uniform
        rollout_length=20,
        time_flat=False,
        use_value_replay=True,
        use_pixel_control=True,
        use_reward_prediction=False,
        rp_reward_threshold=0.3,
        rp_sequence_size=4,
        rp_lambda=0.01,
        model_summary_freq=100,
        episode_summary_freq=5,
        env_render_freq=20,
    )
)


launcher = Launcher(
    cluster_config=cluster_config,
    env_config=env_config,
    trainer_config=trainer_config,
    policy_config=policy_config,
    test_mode=False,
    max_env_steps=100*10**6,
    root_random_seed=0,
    purge_previous=0,  # ask to override previously saved model and logs
    verbose=0  # 0 or 1
)

# Train it:
launcher.run()
