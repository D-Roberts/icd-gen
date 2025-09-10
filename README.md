### WIP - codebase and research under development (energy).

###  Running TODO@DR list [one of the lists]. 

    *-1.  Enerdit dev: . 
        - WIP
        - Comet panels for Enerdit will be here https://www.comet.com/ai-daor/enerdit/view/new/panels

        - Dev step-by-step:
           
            - 2. Dev new loss / dgen / model setup:
            
                - WIP 
        
                - now that model learns, look into what is learning (back to simple for dev new step). 
                
                - Put a test eval psnr with t fixed. why is U going up after a time in training? Looks like a brownian motion with non-zero drift and increasing variance. Is this similar to the next token prediction snowballing error effect by the nature of the problem setup?

                - try to see negative conditional energy by employing the normal approx argument in gao, poole, kingma paper in iclr21.
                
                - What is the model's correct prediction to compare to clean for denoise quality if we are predicting energy (well, space and time scores but so that we get energy). What normalization can I add? Clamping of preds?

                - can I make an "exploding energy" argument a thing.

                - put context back in - with a fully new token from the same distrib (I have an improv right now)

                - to be continued.



    *0. Lens 1. Setting2 [path associations with gamman noise multiplicative]. Code for models on the new patch structure datagen

        [WIP.]

        Comet for patch-group-gamma-denoise-in-context:
        https://www.comet.com/ai-daor/gamma-patch/view/new/panels

        * Rethink how to input the context and prediction for denoise rather than
        classification in the pos association based spatial transformer

        * likely add context embedding here as well not sure yet; 

        * Model 2 - spatialtransformer - added but learning with the denoise task
        with context is - not learning well yet. Debug.

        * reframe problem and input setup.

        * get the associative memories energy
        * WIP

        

    *1. Lens 1. Setting 1. [linear manifold seq] factor in the noise drift code from my other repo where 1 layer doesn't do. I think put back the softmax and not the linear since the 2 layer is with softmax.

    [WIP.]
  
    *3. Lens 1. Setting 1. there might be a bug in the batch loss calculation and/or plotting 
   
    *5. Put in the 2lyaer and train it on drift train - test to see if better generalization than 1 layer. neah this won't work, too hard of a problem. Generate a 3-way interaction data gen example which is pointed to in the birth of transf as a case where need two layer and one will not do.



    Overall:
    *7. Put DAM energies across the board (pun intended).




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
python src/linear/train_linear_denoiser.py --config config/args.yaml
```
Comet at https://www.comet.com/ai-daor/linear-energy/view/new/panels

For gamma noise and patch groups - spatial

```
python src/train_gamma_patch.py --config config/args.yaml
```

For Enerdit:
```
python src/enerdit/dev/train_dev.py
```

Clean up dirs with 
```
./clean.sh
```


Start Comet tracking (Comet is like wandb; used in industry):


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
