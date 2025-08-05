## WIP - codebase under development

### TODO@DR next:
    *1. factor in the noise drift code from my other repo
    *2. there is a bug in the plot loss baselines code
    *3. there might be a bug in the batch loss calculation and/or plotting
    *4. likely a bug in the 2-head archi
    *5. move more params and train choices and archi and data gen options to yaml for cleaner code and faster iteration.
    *6. Refactor archis to: 1. Implement exactly the archi in the theory section. 2. Allow for log of 
    last layer activations and arg to last layer softmax. For the theoretical 2-layer, possibly do not
    linearize softmax.
    *7. Refactor energy and grad fn with torch not numpy.
    *8. Torchify out of numpy pretty much everything up until matplotlib.
    *9. to be cont.
    


### To train:

* Get conda via miniforge (works best for M1).
* Get repo and cd to its root.

```
git clone https://github.com/D-Roberts/icd-gen
```

* Create env (see some comments in env.yml)
```
conda env create -f environment.yml
conda activate icd-gen
```
* Modify args in config/args.yaml 

* Run train
```
python src/train_denoiser.py --config src/config/linear_manifold.yaml
```
Clean up dirs with 
```
./clean.sh
```


Start Comet tracking (Comet is like wandb; used in industry):
https://www.comet.com/ai-daor/icd-gen/view/new/panels

* Track losses and save artifacts results outputs params plots and images per experiment. Clean code and to train properly if I move to more real models and not just toy.

* To be able to create your own experiments, a Comet API Key must be created and placed in the icd-gen/.comet_api file and a Comet account username must be written to icd-gen/.comet_workspace, replacing the credentials I made public in this repo.


Initial code references:
* https://github.com/mattsmart/in-context-denoising
* https://github.com/dtsip/in-context-learning/tree/main/src
* https://github.com/jiachenzhu/DyT
* https://github.com/huggingface/transformers/tree/main/src/transformers
* https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py
