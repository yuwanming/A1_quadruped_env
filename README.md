# A1_quadruped_env

Code for paper:

"Identifying Important Sensory Feedback for Learning Locomotion Skills"

The DRL framework trains A1 quadruped robot to perform five locomotion skills:

* Motor skills trained with full states: balance recovery, trotting, and bounding

* Motor skills trained with key states which are identified through the proposed approach in this paper: balance recovery, trotting, bounding, pacing, and galloping. 

More details can be found in the paper.

## Install dependencies
Recommend to set up conda environment as below:

`conda create -n a1 python=3.6` 

`conda activate a1`

`pip install pybullet==3.0.8`

`pip install -U numpy==1.16`

`conda install tensorflow=1.10 tensorflow-gpu=1.10 cudatoolkit=9.0` (with GPU)

`conda install tensorflow=1.10` (without GPU)

`pip install gym`

`conda install tqdm`

`pip install moviepy`

`conda install -c anaconda scikit-learn`

`conda install matplotlib`

## Run trained policies
### Full-state policies

* Trained full-state policies can be found in the corresponding folder under: `A1_quadruped_env/full_state/SAC/SAC/SAC/record/a1/`

* To run the policies:

    `conda activate a1`
    
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

    `conda activate a1`
    
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



