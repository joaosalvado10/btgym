## <a name="title"></a>Backtrader gym Environment
**Implementation of OpenAI Gym environment for Backtrader backtesting/trading library.**
****
Backtrader is open-source algorithmic trading library:  
GitHub: http://github.com/mementum/backtrader  
Documentation and community:
http://www.backtrader.com/

OpenAI Gym is...,
well, everyone knows Gym:  
GitHub: http://github.com/openai/gym  
Documentation and community:
https://gym.openai.com/
****
### <a name="outline"></a>Outline:
General purpose of this project is to provide gym-integrated framework for
running reinforcement learning experiments 
in [close to] real world algorithmic trading environments.

### [See news and update notes below](#news)

```
DISCLAIMER:
Code presented here is research/development grade.
Can be unstable, buggy, poor performing and is subject to change.

Note that this package is neither out-of-the-box-moneymaker, nor it provides ready-to-converge RL solutions.
Think of it as framework for setting experiments with complex, non stationary, time-series based environments.
I have no idea what kind of algorithm and setup will solve it [if any]. This is work in progress.
```

****
### <a name="contents"></a>Contents
- [Installation](#install)
- [Quickstart](#start)
- [Description](#description)
    - [Problem setting](#problem)
    - [Data sampling approaches](#data)
    - [General notes](#notes)
- [Reference](#reference) 
- [Current issues and limitations](#issues)
- [Roadmap](#roadmap)
- [Update news](#news)
   
   
****
### <a name="install"></a>[Installation](#contents)
It is highly recommended to run BTGym in designated virtual environment.

Clone or copy btgym repository to local disk, cd to it and run: `pip install -e .` to install package and all dependencies:

    git clone https://github.com/Kismuz/btgym.git

    cd btgym

    pip install -e .

To update to latest version::

    cd btgym

    git pull

    pip install --upgrade -e .

##### Note:
BTGym requres Matplotlib version 2.0.2, downgrade your installation if you have version 2.1:

    pip install matplotlib==2.0.2

****
### <a name="start"></a>[Quickstart](#contents)
Making gym environment with all parmeters set to defaults is as simple as:

```python
from btgym import BTgymEnv
 
MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',)
```
Adding more controls may look like:
```python
from btgym import BTgymEnv

MyEnvironment = BTgymEnv(filename='../examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                         episode_duration={'days': 2, 'hours': 23, 'minutes': 55},
                         drawdown_call=50,
                         state_shape=(4,20),
                         port=5555,
                         verbose=1,
                         )
```

##### See more options at [Documentation: Quickstart >>](https://kismuz.github.io/btgym/intro.html#quickstart)

##### and how-to's in [Examples directory >>](./examples).
****
### <a name="description"></a> [General description](#contents)
#### <a name="problem"></a> Problem setting
Consider a discrete-time finite-horizon partially observable Markov decision process for equity/currency trading:
- agent action space is discrete (`buy`, `sell`, `close` [position], `hold` [do nothing]);
- environment is episodic: maximum  episode duration and episode termination conditions
  are set;
- for every timestep of the episode agent is given environment state observation as tensor of last
  m price open/high/low/close values for every equity considered and based on that information is making
  trading decisions.
- agent's goal is to maximize expected cumulative capital;
- classic 'market liquidity' and 'capital impact' assumptions are met.
- environment setup is set close to real trading conditions, including commissions, order execution delays,
  trading calendar etc.

#### <a name="data"></a> Data selection options for backtest agent training:
- random sampling:
  historic price change dataset is divided to training, cross-validation and testing subsets.
  Since agent actions do not influence market, it is possible to randomly sample continuous subset
  of training data for every episode. [Seems to be] most data-efficient method.
  Cross-validation and testing performed later as usual on most "recent" data;
- sequential sampling:
  full dataset is feeded sequentially as if agent is performing real-time trading,
  episode by episode. Most reality-like, least data-efficient, natural non-stationarity remedy.
- sliding time-window sampling:
  mixture of above, episde is sampled randomly from comparatively short time period, sliding from
  furthest to most recent training data. Should be less prone to overfitting than random sampling.
- NOTE: only random sampling is currently implemented.

****
 
### <a name="notes"></a> [Notes](#contents)

 1. There is a choice: where to place most of state observation/reward estimation and prepossessing such as
    featurization, normalization, frame skipping and all other -zation: either to hide it inside environment or to do it
    inside RL algorytm?
    - E.g. while state feature estimators are commonly parts of RL algorithms, reward estimation is often taken
    directly from environment.
    In case of portfolio optimisation reward function can be tricky (not to mention state preprocessing),
    so it is reasonable to make it easyly accessable inside single module for ease of experimenting
    and hyperparameter tuning.
     - BTgym allows to do it both ways: either pass "raw" state observation and do all heavy work inside RL loop
      or put it inside get_state() and get_reward() methods.
    - To mention, it seems reasonable to pass all preprocessing work to server, since it can be done asynchronously
    with agent own computations and thus somehow speed up training.

 2. [state matrix], returned by Environment by default is 2d [n,m] numpy array of floats,
    where n - number of Backtrader Datafeed values: v[-n], v[-n+1], v[-n+2],...,v[0],
    i.e. from n steps back to present step, and every v[i] is itself a vector of m features
    (open, close,...,volume,..., mov.avg., etc.).
    - in case of n=1 process is obviously POMDP. Ensure Markov property by 'frame stacking' or/and
    employing stateful function approximators.
    - When n>1 process [somehow] approaches MDP (by means of Takens' delay embedding theorem).

 3. Why Gym, not Universe VNC environment?
    - At a glance, vnc-type environment should fit algorithmic trading extremely well.
    But to best of my knowledge, OpenAI is yet to publish its "DIY VNC environment" kit. Let's wait.

 4. Why Backtrader library, not Zipline/PyAlgotrader etc.?
    - Those are excellent platforms, but what I really like about Backtrader is clear [to me], flexible  programming logic
    and ease of customisation. You dont't need to do tricks, say, to disable automatic calendar fetching, etc.
    I mean, it's nice feature and making it easy-to-run for trading people but prevents from
    correctly running intraday trading strategies. Since RL-algo-trading is in active research stage, it's impossible to tell
    in advance which setup and logic could do the job. IMO Backtrader is just well suited for this kinds of experiments.
    Besides this framework is being actively maintained.

 5. Why Currency data by default?
    - Obviously environment is data/market agnostic. Backtesting dataset size is what matters.
    Deep Q-value algorithm, most sample efficient among deep RL, take about 1M steps just to lift off.
    1 year 1 minute FX data contains about 300K samples. Feeding dataset consisting of several years of data and
    performing random sampling [arguably]
    makes it realistic to expect algorithm to converge for intra-day or intra-week trading setting (~1500-5000 steps per episode).
    Besides, currency trading holds market liquidity and impact assumptions.
    - That's just preliminary assumption, not proved at all!

 6. Note for backtrader users:
    - There is a shift on meaning 'Backtrader Strategy' in case of reinforcement learning: BtgymStrategy is mostly used for
    technical and service tasks, like data preparation and order executions, while all trading decisions are taken
    by RL agent.

 7. On current implementation: 
    - my commit was to treat backtrader engine as black box and create wrapper using explicitly
    defined and documented methods only. While it is not efficiency-optimised approach, I think
    it is still decent alpha-solution.
****
   
    
### <a name="reference"></a> [Documentation and API Reference >>](https://kismuz.github.io/btgym/)

****
### <a name="issues"></a> [Current issues and limitations:](#title)
- requres Matplotlib version 2.0.2;
- matplotlib backend warning: appears when importing pyplot and using `%matplotlib inline` magic
  before btgym import. It's recommended to import btacktrader and btgym first to ensure proper backend
  choice;
- not tested with Python < 3.5;
- doesn't seem to work correctly under Windows;
- by default, is configured to accept Forex 1 min. data from www.HistData.com;
- only random data sampling is implemented;
- no built-in dataset splitting to training/cv/testing subsets;
- only one equity/currency pair can be traded;
- ~~no 'skip-frames' implementation within environment;~~ done
- ~~no plotting features, except if using pycharm integration observer.~~
    ~~Not sure if it is suited for intraday strategies.~~ [partially] done
- ~~making new environment kills all processes using specified network port. Watch out your jupyter kernels.~~ fixed 

****
### <a name="roadmap"></a> [TODO's and Road Map:](#title)
 - [x] refine logic for parameters applying priority (engine vs strategy vs kwargs vs defaults);
 - [X] API reference;
 - [x] examples;
 - [x] frame-skipping feature;
 - [ ] dataset tr/cv/t approach IN PROGRESS;
 - [x] state rendering;
 - [ ] retrieving results for observers;
 - [x] proper rendering for entire episode;
 - [x] tensorboard integration;
 - [x] multiply agents asynchronous operation feature (e.g for A3C):
 - [x] dedicated data server;
 - [x] multi-modal observation space shape;
 - [x] A3C implementation for BTgym;
 - [x] UNREAL implementation for BTgym;
 - [x] PPO implementation for BTgym;
 - [ ] RL^2 / MAML / DARLA adaptations;
 - [ ] learning from demonstrations;
 - [ ] risk-sensitive agents implementation;
 - [ ] sequential and sliding time-window sampling IN PROGRESS;
 - [ ] multiply instruments trading;
 
 
### <a name="news"></a>[News and updates:](#title)
- 5.12.17: Inner btgym comm. fixes >> speedup ~5%.

- 02.12.17: Basic `sliding time-window train/test` framework implemented via 
            [BTgymSequentialTrial()](https://kismuz.github.io/btgym/btgym.html#btgym.datafeed.BTgymSequentialTrial)
            class.
            
- 29.11.17: Basic meta-learning RL^2 functionality implemented.
    - See [Trial_Iterator Class](https://kismuz.github.io/btgym/btgym.html#btgym.datafeed.BTgymRandomTrial) and
    [RL^2 policy](https://kismuz.github.io/btgym/btgym.research.html#btgym.research.policy_rl2.AacRL2Policy)
    for description.
    - Effectiveness is not tested yet, examples are to follow.

- 24.11.17: A3C/UNREAL finally adapted to work with BTGym environments.
    - Examples with synthetic simple data(sine wawe) and historic financial data added,
      see [examples directory](./examples/);
    - Results on potential-based functions reward shaping in `/research/DevStartegy_4_6`;
    - Work on Sequential/random Trials Data iterators (kind of sliding time-window) in progress,
      start approaching the toughest part: non-stationarity battle is ahead.

- 14.11.17: BaseAAC framework refraction; added per worker batch-training option and LSTM time_flatten option; Atari
            examples updated; see [Documentation](https://kismuz.github.io/btgym/) for details.
            
- 30.10.17: Major update, some backward incompatibility:
    - BTGym now can be thougt as two-part package: one is environment itself and the other one is
      RL algoritms tuned for solving algo-trading tasks. Some basic work on shaping of later is done. Three advantage
      actor-critic style algorithms are implemented: A3C itself, it's UNREAL extension and PPO. Core logic of these seems
      to be implemented correctly but further extensive BTGym-tuning is ahead.
      For now one can check [atari tests](./examples/atari_tests).
    - Finally, basic [documentation and API reference](https://kismuz.github.io/btgym/) is now available.

- 27.09.17: A3C [test_4.2](./examples/a3c/a3c_test_4_2_no_feature_signal_conv1d.ipynb) added:
    - some progress on estimator architecture search, state and reward shaping;

- 22.09.17: A3C [test_4](./examples/a3c/a3c_test_4_sma_bank_features.ipynb) added:
    - passing train convergence test on small (1 month) dataset of EURUSD 1-minute bar data; 

- 20.09.17: A3C optimised sine-wave test added [here.](./examples/a3c/a3c_reject_test_3_sine_conv1d_sma_log_grad.ipynb)
    - This notebook presents some basic ideas on state presentation, reward shaping,
      model architecture and hyperparameters choice.
      With those tweaks sine-wave sanity test is converging faster and with greater stability.

- 31.08.17: Basic implementation of A3C algorithm is done and moved inside BTgym package.
    - algorithm logic consistency tests are passed;
    - still work in early stage, experiments with obs. state features and policy estimator architecture ahead;
    - check out [`examples/a3c`](./examples/a3c) directory.

- 23.08.17: `filename` arg in environment/dataset specification now can be list of csv files.
    - handy for bigger dataset creation;
    - data from all files are concatenated and sampled uniformly;
    - no record duplication and format consistency checks preformed.

- 21.08.17: UPDATE: BTgym is now using multi-modal observation space.
     - space used is simple extension of gym: `DictSpace(gym.Space)` - dictionary (not nested yet) of core gym spaces.
     - defined in `btgym/spaces.py`.
     - `raw_state` is default Box space of OHLC prices. Subclass BTgymStrategy and override `get_state()` method to
            compute alll parts of env. observation.
     - rendering can now be performed for avery entry in observation dictionary as long as it is Box ranked <=3
            and same key is passed in reneder_modes kwarg of environment.
            'Agent' mode renamed to 'state'. See updated examples.
 

- 07.08.17: BTgym is now optimized for asynchronous operation with multiply environment instances.
     - dedicated data_server is used for dataset management;
     - improved overall internal network connection stability and error handling;
     - see example `async_btgym_workers.ipynb` in [`examples`](./examples) directory.

- 15.07.17: UPDATE, BACKWARD INCOMPATIBILITY: now state observation can be tensor of any rank.
     - Consequently, dim. ordering convention has changed to ensure compatibility with 
            existing tf models: time embedding is first dimension from now on, e.g. state
            with shape (30, 20, 4) is 30x steps time embedded with 20 features and 4 'channels'.
            For the sake of 2d visualisation only one 'cannel' can be rendered, can be
            chosen by setting env. kwarg `render_agent_channel=0`;
     - examples are updated;
     - better now than later.

- 11.07.17: Rendering battle continues: improved stability while low in memory,
            added environment kwarg `render_enabled=True`; when set to `False`
             - all renderings are disabled. Can help with performance.

- 5.07.17:  Tensorboard monitoring wrapper added; pyplot memory leak fixed.

- 30.06.17: EXAMPLES updated with 'Setting up: full throttle' how-to. 

- 29.06.17: UPGRADE: be sure to run `pip install --upgrade -e .` 
    - major rendering rebuild: updated with modes: `human`, `agent`, `episode`;
      render process now performed by server and returned to environment as `rgb numpy array`.
      Pictures can be shown either via matplolib or as pillow.Image(preferred).
    - 'Rendering HowTo' added, 'Basic Settings' example updated.
    - internal changes: env. state divided on `raw_state`  - price data,
      and `state` - featurized representation. `_get_raw_state()` method added to strategy.
    - new packages requirements: `matplotlib` and `pillow`.

- 25.06.17:
  Basic rendering implemented. 

- 23.06.17:
  alpha 0.0.4:
  added skip-frame feature,
  redefined parameters inheritance logic,
  refined overall stability;
  
- 17.06.17:
  first working alpha v0.0.2.
 
 
