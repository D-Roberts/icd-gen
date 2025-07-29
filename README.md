WIP - codebase under development

```
python train_denoiser.py --config conf/linear_denoise.yaml
```
from within src. Though the environment needs to be set up first, instructions mostly in new_env_todo.yml and then
setup a conda env which works best on M1 with miniforge.

Modify args in yaml.

See stuff in wandb you need to have an account and put it like denisa-roberts for me in yaml.

Initial code references:
https://github.com/mattsmart/in-context-denoising
https://github.com/dtsip/in-context-learning/tree/main/src
https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py

Start Comet tracking (like wandb, used in industry):
https://www.comet.com/ai-daor/icd-gen/view/new/panels