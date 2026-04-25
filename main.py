import logging

from python_impl.verifier.torch_verifier import verify_torch
from python_impl.basic.torch_brief import brief_torch
from python_impl.toy_model.torch_toy_model import toy_model_torch
from python_impl.train.torch_train import train_torch
from python_impl.load_model.torch_load_model import load_model_torch
from python_impl.fine_tuning.torch_fine_tuning import fine_tuning_torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")

logger.info("Run torch version verifier")
verify_torch()
logger.info("Run torch basic examples")
brief_torch()
logger.info("Run toy model generation example")
toy_model_torch()
logger.info("Run toy model training example")
#train_torch()
logger.info("Load public model to toy model")
#load_model_torch()
logger.info("Fine-tuning")
fine_tuning_torch()