### WIP - codebase under development.

### TODO@DR running list 

    *-1. Code up the Enerdit: priority 1. 
        - WIP
        - Comet panels for Enerdit will be here https://www.comet.com/ai-daor/enerdit/view/new/panels
        - Start debug step-by-step:
            - 1. Debug archi with some known loss. [WIP]
                - add some init to begin with
                - use MAE on space score and clean label only [done]
                - retain only emebd and linears [done]
                - next - need to make debug dataset small and easy a la Karpathy but not quite (one batch is too little)
                [done]
            - 2. Dev new loss.
                - need to replace the gamma noise right now with the gaussian
                - not sure what the t schedule to be for the sequence 
                gaussian. Right now the gamma is tied to the structure config
                with the groups. Sampling t from tmin tmax won't preserve that.
                Not sure if that's important or not.
                - clip clean to [0, 1]
                - debug both space and loss expression [done first pass - better shape technically but not learning the current datagen (while
                MAE was on just the clean and noisy query)]
                - be sure extracting the right non-padding portion of query
                and label too
                - Priority 1: the right noise schedule to get the t accross
                the sequence
                
           

    *0. Code for models on the new patch structure datagen (priority 2)

        *reconsider if sinusoidal or time seq pos embed (now sinusoidal)
        *reconsider if to pad target to 2*patch dim to match fused seq or to dim
        since need to project it from embed dim anyways

        * reconsider loss. Now MSE def not appropriate for gamma noise. MAE option now. Should I make a new loss? Distribution level comparison is best. How would I get an energy-based loss here? 

        * also partition of indeces itneresting. Maybe leverage it more. Make it random.

        * MAE works better than MSE as theory suggests

        * Not better with the large GPT2 frozen backbone on the small set. Haven't tested the ViT yet.

        * nlmeans per one-shot does better than a learned one-layer attention with pos embeds and input output projections

        * When mimic an in-context where partitions generated per batch - model perf goes down.

        * save the datasets after generating and load from disk- it gets slow.
        
        * do some input normalization

    *1. factor in the noise drift code from my other repo [the icd first task]
    *2. there is a bug in the plot loss baselines code [the icd first task]
    *3. there might be a bug in the batch loss calculation and/or plotting [the icd first task]
    *4. likely a bug in the 2-head archi [the icd first task]
    *6. Refactor linear and spatial groups 1 and 2 archis to implement exactly the archis in the theory section. 
    *7. Refactor DAM energy and grad fn with torch not numpy. Also - debug, result should not be a matrix.
    *8. Torchify out of numpy pretty much everything up until matplotlib. Use torch efficient implementations for functions where available, for instance Jacobians(torch jacrev) if I need them.
   
    


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
```
python src/train_denoiser.py --config config/args.yaml
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
