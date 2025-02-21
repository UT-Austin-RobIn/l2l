# L2L
Note: The task names between the paper and codebase are slightly different. We have three simulation tasks -- **kitchen**, **walled** and **two_arm** which correspond to **cooking**, **walls** and **assembly** tasks in the paper respectively.

## Setup
### Installation
```
conda create --name l2l python==3.10
conda activate l2l
git clone --recursive https://github.com/ShivinDass/l2l.git
cd l2l && git submodule update --init --recursive
pip install -r requirements.txt
pip install -e stable-baselines3/.
pip install -e .
```

Install robosuite,
```
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite 
git checkout 48c1b8a6c077d04399a00db05694d7f9f876ffc9
pip install -e .
```

For the two_arm task we use some assets from [mimicgen]() so optionally set that up as well,
```
git clone https://github.com/NVlabs/mimicgen.git
cd mimicgen
pip install -e .
``` 


## Usage
Our proposed solution DISaM works in two phases,
1. Phase 1: Train an Information-Receiving (IR) policy using imitation learning. ([pretrained ckpts](https://utexas.box.com/s/i92e66vwd1985xmkhnrsput0bj0xb70h))
2. Phase 2: Freeze the pretrained IR policy and train the Information-Seeking (IS) policy using RL. ([pretrained ckpts](https://utexas.box.com/s/jwglzicuax5rx516soe4r9psvyc05zzf))

Following we provide instructions for the **walled** task but they can be appropriately modified for **kitchen** and **two_arm** tasks.
### Phase 1: Imitation Learning (IR)
1. Download the [data](https://utexas.box.com/s/zui5pynnbhi4a07p1ah29vmak7v7rw69) (ex. skill_walled_oh_n200.h5) and change the data path in the [IR config]() file.
2. ```
    python l2l/scripts/train_il.py \
    --config l2l/config/il/bc_ce_walled_multi_stage_config.py \
    --exp_name il_walled
    ```

### Phase 2: Reinforcement Learning (IS)
1. Change the path in the [IS config]() file to point to the trained ckpt from phase 1 or use the provided [pretrained ckpts](https://utexas.box.com/s/i92e66vwd1985xmkhnrsput0bj0xb70h) (ex. walled/weights/weights_ep15.pth).
2. ```
    python l2l/scripts/dual_optimization.py \
    --config l2l/config/dual/robosuite/skill_walled_multi_stage/walled_multi_stage_action_dual_config.py \
    --exp_name disam_walled
    ```


### Evaluation
```
python l2l/scripts/final_eval_dual.py --env <task-name> --info_step_break 3 --ckpt path/to/IS_ckpt.zip --n_rollouts 50
```
Where ```<task-name>``` is one of **kitchen**, **walled** or **two_arm** and set --ckpt to the trained RL ckpt path from phase 2 or try one of the [pretrained ckpts](https://utexas.box.com/s/jwglzicuax5rx516soe4r9psvyc05zzf) (ex. walled/disam_walled/epoch_xx/rl_model_xxxx_steps).