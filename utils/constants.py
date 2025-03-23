import os

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


MODEL_GCLOUD_BUCKET_PATH = "https://storage.googleapis.com/stanford_neuroai_models"

_cache_prefix = os.getenv("CACHE", ".cache")
MODEL_LOCAL_CACHE_PATH = f"{_cache_prefix}/stanford_neuroai_models"
