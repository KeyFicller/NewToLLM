import argparse

from python_impl.verifier.torch_verifier import verify_torch
from python_impl.basic.torch_brief import brief_torch
from python_impl.toy_model.torch_toy_model import toy_model_torch
from python_impl.train.torch_train import train_torch
from python_impl.load_model.torch_load_model import load_model_torch
from python_impl.fine_tuning.torch_fine_tuning import fine_tuning_torch

def run_verify():
    print("--- Run torch version verifier ---")
    verify_torch()


def run_brief():
    print("--- Run torch basic examples ---")
    brief_torch()


def run_toy_model():
    print("--- Run toy model generation example ---")
    toy_model_torch()


def run_train():
    print("--- Run toy model training example ---")
    train_torch()


def run_load_model():
    print("--- Load public model to toy model ---")
    load_model_torch()


def run_fine_tuning():
    print("--- Run torch fine-tuning ---")
    fine_tuning_torch()


def parse_args():
    parser = argparse.ArgumentParser(description="LLM_from_scratch entrypoint")
    parser.add_argument(
        "task",
        nargs="?",
        default="default",
        choices=[
            "default",
            "verify",
            "brief",
            "toy_model",
            "train",
            "load_model",
            "fine_tuning",
            "all",
        ],
        help="Task to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "verify":
        run_verify()
    elif args.task == "brief":
        run_brief()
    elif args.task == "toy_model":
        run_toy_model()
    elif args.task == "train":
        run_train()
    elif args.task == "load_model":
        run_load_model()
    elif args.task == "fine_tuning":
        run_fine_tuning()
    elif args.task == "all" | args.task == "default":
        run_verify()
        run_brief()
        run_toy_model()
        run_train()
        run_load_model()
        run_fine_tuning()


if __name__ == "__main__":
    main()