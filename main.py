from python_impl.verifier.torch_verifier import verify_torch
from python_impl.basic.torch_brief import brief_torch
from python_impl.toy_model.torch_toy_model import toy_model_torch
from python_impl.train.torch_train import train_torch
from python_impl.load_model.torch_load_model import load_model_torch
from python_impl.fine_tuning.torch_fine_tuning import fine_tuning_torch

# This category includes some torch environment info
print("--- Run torch version verifier ---")
verify_torch()

# This category includes some torch basic concepts
print("--- Run torch basic examples ---")
brief_torch()

#logger.info("Run toy model generation example")
#toy_model_torch()
#logger.info("Run toy model training example")
#train_torch()
#logger.info("Load public model to toy model")
#load_model_torch()
# logger.info("Fine-tuning")
# fine_tuning_torch()