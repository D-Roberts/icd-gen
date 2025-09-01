### WIP - codebase under development.

### TODO@DR running list.

    *-1.  Enerdit dev: . 
        - WIP
        - Comet panels for Enerdit will be here https://www.comet.com/ai-daor/enerdit/view/new/panels

        - Dev step-by-step:
           
            - 2. Dev new loss / dgen / model setup to enable learning.
            
                - WIP 
        
                - now that model learns, look into what is learning (back to simple for dev new step). 
                
                - Put a test eval psnr with t fixed. why is U going up after a time in training? Looks like a brownian motion with non-zero drift and exploding variance. 
                
                - What is the model's correct prediction to compare to clean for denoise quality if we are predicting energy (well, space and time scores but so that we get energy). What normalization can I add? Clamping of preds?

                - put context back in - with a fully new token from the same distrib (I have an improv right now)

                - to be continued.



    *0. Lens 1. Setting2. Code for models on the new patch structure datagen

        Comet for patch-group-gamma-denoise-in-context:
        https://www.comet.com/ai-daor/gamma-patch/view/new/panels

        * Rethink how the groups and group indeces are generated.

        * Model 2 - is work in progress.

        *reconsider if to pad target to 2*patch dim to match fused seq or to dim
        since need to project it from embed dim anyways. For pred and loss- slice out zeros. [not sure yet how to best do this]

        * compare MAE with one other loss (no point in MSE). 
        

    *1.Lens 1. Setting 1. factor in the noise drift code from my other repo 
    *2. Lens 1. Setting 1.  there is a bug in the plot loss baselines code 
    *3. Lens 1. Setting 1. there might be a bug in the batch loss calculation and/or plotting 
    #4. Lens 1. Setting 1. There might be a bug in the 2-attn heads repo.


    Overall:
    *7. Refactor DAM energy and grad fn with torch not numpy. Also - debug, result should not be a matrix. 
    
    *8. Input normalizations accross the board.


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

* Run train from root of icd-gen dir

For linear datagen
```
python src/train_denoiser.py --config config/args.yaml
```
For gamma noise and patch groups - spatial

```
python src/train_gamma_patch.py --config config/args.yaml
```

Clean up dirs with 
```
./clean.sh
```


Start Comet tracking (Comet is like wandb; used in industry):
https://www.comet.com/ai-daor/icd-gen/view/new/panels

* Track losses and save artifacts results outputs params plots and images per experiment. Clean code and to train properly when I move to more real models and not just toy.

* To be able to create your own experiments, a Comet API Key must be created and placed in the icd-gen/.comet_api file and a Comet account username must be written to icd-gen/.comet_workspace, replacing the credentials I made public in this repo.


Initial code references:
* https://github.com/smarter-vlm/smarter/blob/main/main_reasoner.py
* https://github.com/mattsmart/in-context-denoising
* https://github.com/dtsip/in-context-learning/tree/main/src
* https://github.com/jiachenzhu/DyT
* https://github.com/huggingface/transformers/tree/main/src/transformers
* https://github.com/LabForComputationalVision/memorization_generalization_in_diffusion_models/tree/main
* VIT learn spatial struct https://proceedings.neurips.cc/paper_files/paper/2022/hash/f69707de866eb0805683d3521756b73f-Abstract-Conference.html
* https://github.com/facebookresearch/DiT/blob/main/models.py
* https://github.com/Zhendong-Wang/Patch-Diffusion/blob/main/training/augment.py
* https://github.com/yang-song/score_sde/blob/main/losses.py
