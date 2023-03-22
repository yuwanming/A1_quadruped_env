# A1_quadruped_env

Code for paper:

"Identifying Important Sensory Feedback for Learning Locomotion Skills"

The DRL framework trains A1 quadruped robot to perform five locomotion skills:

* Motor skills trained with full states: balance recovery, trotting, and bounding.

* Motor skills trained with key states which are identified through the proposed approach in this paper: balance recovery, trotting, bounding, pacing, and galloping. 

More details can be found in the paper.

## Install dependencies
The code has been tested in Ubuntu 16.04, 18.04 and 20.04. Recommend to install Anaconda (link for install instructions [here](https://docs.anaconda.com/anaconda/install/linux/))

After installing Anaconda, set up conda environment and install dependencies as below:

`./run_create_conda_env.sh` 

## Run trained policies
### Full-state policies

* Trained full-state policies can be found in the corresponding folder under: `A1_quadruped_env/full_state/SAC/SAC/SAC/record/a1/`

* To run the policies:
    
    `cd A1_quadruped_env/full_state/SAC/SAC/`
    
    `python run_a1_3D_SAC_standup.py`
    
    `python run_a1_3D_SAC_trot.py`
    
    `python run_a1_3D_SAC_bound.py`

* To train new policies with full states under the same folder:

    `python train_a1_3D_SAC_standup.py`
    
    `python train_a1_3D_SAC_trot.py`
    
    `python train_a1_3D_SAC_bound.py`

### Key-state policies
* Trained key-state policies can be found in the corresponding folder under:
`A1_quadruped_env/key_state/SAC/SAC/SAC/record/a1/`

* To run the policies:
    
    `cd A1_quadruped_env/key_state/SAC/SAC/`
    
    `python run_a1_3D_SAC_standup.py`
    
    `python run_a1_3D_SAC_trot.py`
    
    `python run_a1_3D_SAC_bound.py`
    
    `python run_a1_3D_SAC_pace.py`
    
    `python run_a1_3D_SAC_gallop.py`

* To train new policies with key states under the same folder:

    `python train_a1_3D_SAC_standup.py`
    
    `python train_a1_3D_SAC_trot.py`
    
    `python train_a1_3D_SAC_bound.py`
    
    `python train_a1_3D_SAC_pace.py`
    
    `python train_a1_3D_SAC_gallop.py`



