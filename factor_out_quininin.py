# my_script.py
import argparse
import yaml

parser = argparse.ArgumentParser(description="My application description.")
parser.add_argument("--config-file", help="Path to YAML config file")
parser.add_argument("--model", default="default_value")

args = parser.parse_args()

if args.config_file:
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
        parser.set_defaults(**config)
    args = parser.parse_args() # Reload arguments to apply YAML values


print(f"My option: {args.model["family"]}")
