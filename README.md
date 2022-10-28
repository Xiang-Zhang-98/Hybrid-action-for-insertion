This repo suppports our ICRA 22 submission ["Learning Insertion Primitives with Discrete-Continuous Hybrid Action Space for Robotic Assembly tasks"](https://ieeexplore.ieee.org/abstract/document/9811973?casa_token=DRABv1TPk2IAAAAA:RGcsWWlMX4rymzsxuYR_3tl4S-AFUX2l-pns4H_syGcHRj8AFtm1O1kPaBU4yCS5DnRqN2fQ0g)
This repo is developed based on the code from MP-DQN. [paper](https://arxiv.org/abs/1905.04388) [code](https://github.com/cycraig/MP-DQN)
## Installation
### Setup MuJoCo
1. Download the MuJoCo version 2.0 binaries
    * Download Mujoco200 and activation key from https://www.roboti.us/download.html https://www.roboti.us/license.html
2. Extract the downloaded `mujoco200` directory into `~/.mujoco/mujoco200`
3. Append the following lines to `~/.bashrc` and `source ~/.bashrc`
 ```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/robodemo/.mujoco/mujoco200/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```
### Create conda environment
```bash
conda create --name insertion python=3.7 pip
conda activate insertion
pip install -r requirements.txt
cd pegibhole_env
pip install -e .
cd ..
```

## Training
```bash
python TS-MP-DQN/run_peginhole_3prms.py # train peg-in-hole policy
python TS-MP-DQN/run_policy_3prms.py --policy_file /proposed_square_peg1/19999 # run policy
python TS-MP-DQN/run_transferlearning_3prms.py --episodes 5000 --loadingpath "/proposed_square_peg1/19999" --pegshape "pentagon" --fintune_start_episode 1000 # transfer learning
```

## Citing
If this repository has helped your research, please cite the following:

```BibTeX
@inproceedings{zhang2022learning,
  title={Learning insertion primitives with discrete-continuous hybrid action space for robotic assembly tasks},
  author={Zhang, Xiang and Jin, Shiyu and Wang, Changhao and Zhu, Xinghao and Tomizuka, Masayoshi},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)},
  pages={9881--9887},
  year={2022},
  organization={IEEE}
}
```