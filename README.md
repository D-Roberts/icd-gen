In context denoising tasks (perhaps including masking and inpainting and superesolution as tasks to be sampled from at inference time based on prompt pairs) from a generative transformers energy perspective as a one step toward superintelligence.

In-context- dual score matching and energy objective for the generative approach.


```
python train_denoiser.py --config conf/linear_denoise.yaml
```
from within src

Modify args in yaml.

See stuff in wandb you need to have an account and put it like denisa-roberts for me in yaml.

Initial code references:
https://github.com/mattsmart/in-context-denoising
https://github.com/dtsip/in-context-learning/tree/main/src
https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py

Initial wandb
https://wandb.ai/denisa-roberts/in-context-denoising?nw=nwuserdenisaroberts