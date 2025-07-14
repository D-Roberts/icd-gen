import os
from random import randint
import uuid

#from quinine import QuinineArgumentParser
# TODO@DR replace with arparse or similar
import argparse

from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler

# from schema import schema
from models import build_model_toy

import wandb

torch.backends.cudnn.benchmark = True



# AVAIL_GPUS = min(1, torch.cuda.device_count())
if not torch.backends.mps.is_available():
    print('\nMPS device not found.')
    mps_device = None
     
if torch.backends.mps.is_available():
        device = torch.device("mps")
        mps_device = torch.device("mps")
        x = torch.ones(1, device=device)
        print('\nCheck M1 chip:', x)
elif torch.cuda.is_available():
        device = torch.device("cuda:0")
else:
        device = "cpu"
print('device selected:', device)

#device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Available GPUs {AVAIL_GPUS} and current device {device}")


def train_step(model, xs, ys, optimizer, loss_func):
    optimizer.zero_grad()
    output = model(xs, ys)
    loss = loss_func(output, ys)
    loss.backward()
    optimizer.step()
    # TODO @DR: notice no clip_grad_norm - should we have it?
    # TODO @DR: notice no scheduler - should we have it?
    # TODO @DR: no notion of validation - ? or early stopping?

    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args={}):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # curriculum = Curriculum({})

    starting_step = 0
    state_path = os.path.join(args["out_dir"], "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        print("state dict?", state["model_state_dict"])

        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        # for i in range(state["train_step"] + 1):
        #     curriculum.update()

    n_dims = 20
    bsize = 64
    data_sampler = get_data_sampler("gaussian", n_dims=n_dims)
    task_sampler = get_task_sampler(
        "linear_regression",
        n_dims,
        bsize,
        # num_tasks=args.training.num_tasks, # not sure but does not seem to be populated in configs
        num_tasks=None
    )
    pbar = tqdm(range(0, 5000)) # start_step to num train_steps

    num_training_examples = None # does not seem to be populated i config

    for i in pbar: #num train steps
        data_sampler_args = {}
        task_sampler_args = {}

        
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        # data_sampler.sample_xs
        xs = data_sampler.sample_xs(
            41, #this arg is n_points in sampler, # curriculum.n_points, # args.points.start in config 
            # or Kcur in appendix A which grows to 41 for linear regression starting from 11
            # is this seq len?
            bsize,
            20, # would be dim truncated in curriculum
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs) # this is like the ground truth / label generated on the fly with the task

        loss_func = task.get_training_metric() #MSE for lin reg

        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, loss_func)
        #ys comes from the task; generated in training on the fly

        point_wise_tags = list(range(41)) # this was curriculum.n_points
        point_wise_loss_func = task.get_metric() # squared error
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        # average over the curriculum but here no curriculum but can keep the code
        # this baseline loss is for the specific task , here in toy is Lin Reg
        baseline_loss = (
            sum(
                max(20 - ii, 0) # no n_dim truncated; 20 is always in this work.
                for ii in range(41) # this would be curriculum.n_points so 41 for Lin reg
            )
            / 41 #curriculum.n_points
        )

        log_every_steps = 10 # here in toy log all steps; not sure what test_run means
        if i % log_every_steps == 0 and not args["test_run"]:
            wandb.log(
                {
                    "overall_loss": loss, # for the in-context transformer
                    "excess_loss": loss / baseline_loss, #compared to the lin reg
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": 41, #curriculum n_points
                    "n_dims": 20,
                },
                step=i,
            )

        # curriculum.update() #no curriculum; this would have increased dtrunc by 1, k by 2 every 2K steps
        
        
        save_every = 1000
        pbar.set_description(f"loss {loss}")

        # this is save_every_steps config
        if save_every == 1000 and not args["test_run"]:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)
            # print("saved training state dict", training_state)

        # this is keep_every_steps config
        keep_every_steps = 10000
        if (
            keep_every_steps > 0
            and i % keep_every_steps == 0
            and not args["test_run"]
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args["out_dir"], f"model_{i}.pt"))


def main(args):
    args["test_run"] = None
    if args["test_run"]:
        curriculum_args = args.training.curriculum
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 5000
    else:
        wandb.init(
            dir="../models/linear_regression",
            project="linear_regression_toy",
            entity="denisa-roberts",
            config="",
            notes="",
            name="",
            resume=True,
        )

    model = build_model_toy() #n_embed is 128 still I think
    
    model.to(device)
    model.train()

    train(model, args)


    # if not args["test_run"]:
    #     _ = get_run_metrics(args["out_dir"])  # precompute metrics for eval


if __name__ == "__main__":
    # parser = QuinineArgumentParser(schema=schema)
    # parser = argparse.ArgumentParser()
    # args = parser.parse_quinfig()
    # args = parser.parse_args()
    args = {}

    # assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")
    # args.model.family = "gpt2"
    args["model"] = "gpt2"
    args["test_run"] = None

    if not args["test_run"]:
        # run_id = args.training.resume_id
        run_id = None
        if run_id is None:
            run_id = str(uuid.uuid4())

        print("run_id is", run_id)

        out_dir = os.path.join("../models/linear_regression", run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args["out_dir"] = out_dir
        print("out_dir", out_dir)
        

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args, yaml_file, default_flow_style=False)

    main(args)
    
# All GPT2 model parameters:
# Layer: wte.weight, Shape: torch.Size([50257, 64])
# Layer: wpe.weight, Shape: torch.Size([82, 64])
# Layer: h.0.ln_1.weight, Shape: torch.Size([64])
# Layer: h.0.ln_1.bias, Shape: torch.Size([64])
# Layer: h.0.attn.c_attn.weight, Shape: torch.Size([64, 192])
# Layer: h.0.attn.c_attn.bias, Shape: torch.Size([192])
# Layer: h.0.attn.c_proj.weight, Shape: torch.Size([64, 64])
# Layer: h.0.attn.c_proj.bias, Shape: torch.Size([64])
# Layer: h.0.ln_2.weight, Shape: torch.Size([64])
# Layer: h.0.ln_2.bias, Shape: torch.Size([64])
# Layer: h.0.mlp.c_fc.weight, Shape: torch.Size([64, 256])
# Layer: h.0.mlp.c_fc.bias, Shape: torch.Size([256])
# Layer: h.0.mlp.c_proj.weight, Shape: torch.Size([256, 64])
# Layer: h.0.mlp.c_proj.bias, Shape: torch.Size([64])
# Layer: ln_f.weight, Shape: torch.Size([64])
# Layer: ln_f.bias, Shape: torch.Size([64])

#TODO @DR - doesn't seem that it is learning