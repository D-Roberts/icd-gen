WIP - codebase under development
 TODO@DR  move more params and train choices and archi and data gen options to yaml for clean code and fast iteration

```
python train_denoiser.py --config config/linear_manifold.yaml
```
from within src. Though the environment needs to be set up first, instructions mostly in new_env_todo.yml and then
setup a conda env which works best on M1 with miniforge.

Modify args in yaml.


Initial code references:
https://github.com/mattsmart/in-context-denoising
https://github.com/dtsip/in-context-learning/tree/main/src
https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py

Start Comet tracking (like wandb, used in industry):
https://www.comet.com/ai-daor/icd-gen/view/new/panels

To be able to create personal experiments, a Comet API Key must be created and placed in the smarter/.comet_token file and a Comet account username must be written to smarter/.comet_workspace (from your CometML account), replacing the public one.