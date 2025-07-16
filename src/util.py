
import datetime
import numpy as np

import os 
import pickle #must we? TODO: get rid of the pickle later
import torch 
# keep these until I understand exactly how the vis works

def run_subdir_setup(dir_runs, run_subfolder=None, timedir_override=None, minimal_mode=False):
    """
    Create a new directory for the run, and save the model trajectory and dataset there

    Structure:
        /runs-dir/
            /dir_base (timestamped run folder)
                /model_checkpoints/...
                /vis/...
                /data_for_replot/...
                /model_end.pth
                /training_dataset_split.npz
                /runinfo.txt
    """
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%I.%M.%S%p")
    experiment_dir = dir_runs

    if timedir_override is not None:
        time_folder = timedir_override
    else:
        time_folder = current_time

    if run_subfolder is None:
        dir_current_run = experiment_dir + os.sep + time_folder
    else:
        if os.path.isabs(run_subfolder):
            dir_current_run = run_subfolder + os.sep + time_folder
        else:
            dir_current_run = experiment_dir + os.sep + run_subfolder + os.sep + time_folder

    # make subfolders in the timestamped run directory:
    dir_checkpoints = os.path.join(dir_current_run, "model_checkpoints")
    dir_vis = os.path.join(dir_current_run, "vis")
    dir_data_for_replot = os.path.join(dir_current_run, "data_for_replot")

    if minimal_mode:
        dir_list = [dir_runs, dir_current_run]
    else:
        dir_list = [dir_runs, dir_current_run, dir_checkpoints, dir_vis, dir_data_for_replot]
    for dirs in dir_list:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    # io path storage to pass around
    io_dict = {'dir_base': dir_current_run,
               'dir_checkpoints': dir_checkpoints,
               'dir_vis': dir_vis,
               'dir_data_for_replot': dir_data_for_replot,
               'runinfo': dir_current_run + os.sep + 'runinfo.txt'}

    # make minimal run_info settings file with first line as the base output dir
    runinfo_append(io_dict, ['dir_base, %s' % dir_current_run])

    return io_dict


def runinfo_append(io_dict, info_list, multi=False):
    """
    append to metadata file storing parameters for the run
    """
    # multi: list of list flag
    if multi:
        with open(io_dict['runinfo'], 'a') as runinfo:
            for line in info_list:
                runinfo.write('\n'.join(line))
    else:
        with open(io_dict['runinfo'], 'a') as runinfo:
            runinfo.write('\n'.join(info_list))
            runinfo.write('\n')
    return


def load_runinfo_from_rundir(dir_run, model_prefix=""):
    """
    Load runinfo.txt from a given run directory
    """
    runinfo_fpath = dir_run + os.sep + model_prefix + "runinfo.txt"
    with open(runinfo_fpath, "r") as runinfo:
        runinfo_lines = runinfo.readlines()

    # step: convert runinfo to a dictionary and return it
    runinfo_dict = {}
    for line in runinfo_lines:
        if line.split(",")[0].strip() == "scheduler":
            key = line.split(",")[0]
            val = ",".join(line.split(",")[1:]).strip()
            val = eval(val)
            assert isinstance(val, dict)
        else:
            key, val = line.split(",")
            key, val = key.strip(), val.strip()
        runinfo_dict[key] = val

    # handle potentially ambiguous key -> value types
    # - style_subspace_dimensions (int or str)
    # - seed_dataset (None or int)
    # - could have diff number of keys relating to gradient descent e.g. adam_lr, sgd_lr, etc. -- keep as str
    for key in [
        "epochs",
        "batch_size",
        "dim_n",
        "context_len",
        "train_plus_test_size",
        "full_loss_sample_interval",
        "context_examples_per_W",
        "samples_per_context_example",
        "num_W_in_dataset",
    ]:
        runinfo_dict[key] = int(runinfo_dict[key])
    for key in ["style_subspace_dimensions"]:
        if runinfo_dict[key] != "full":
            runinfo_dict[key] = int(runinfo_dict[key])
    for key in ["sigma2_corruption", "sigma2_pure_context", "test_ratio"]:
        runinfo_dict[key] = float(runinfo_dict[key])
    for key in ["style_corruption_orthog", "style_origin_subspace"]:
        runinfo_dict[key] = (
            runinfo_dict[key] == "True"
        )  # if True, the bool val is True, else False

