## WIP - codebase under development

### TODO@DR next:
    *1. factor in the noise drift code from the other repo
    *2. there is a bug in the plot loss baselines code
    *3. there might be a bug in the batch loss calculation and/or plotting
    *4. move more params and train choices and archi and data gen options to yaml for cleaner code and faster iteration.
    

 
To train:
```
python src/train_denoiser.py --config src/config/linear_manifold.yaml
```
Though the environment needs to be set up first, instructions mostly in new_env_todo.yml and then
setup a conda env which works best on M1 with miniforge.

Modify args in yaml.


Initial code references:
* https://github.com/mattsmart/in-context-denoising
* https://github.com/dtsip/in-context-learning/tree/main/src
* https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py

Start Comet tracking (Comet is like wandb; used in industry):
https://www.comet.com/ai-daor/icd-gen/view/new/panels

* Track losses and save artifacts results outputs params plots and images per experiment. Clean code and "watch the training live for guiding choices" if I move to more real models and not just toy.

To be able to create your own experiments, a Comet API Key must be created and placed in the icd-gen/src/.comet_token file and a Comet account username must be written to icd-gen/src/.comet_workspace, replacing the credentials I made public in this repo.
