import argparse
import os

from utils import dist_logging, utils

logger = dist_logging.get_logger(__name__)

# Base URL for downloading files
base_url = "https://storage.googleapis.com/tapvid_pickles"
available_datasets = {
    "davis": ["tapvid_davis.pkl"],
    "kinetics": ["tapvid_kinetics_sampled.pkl"],
    "all": ["tapvid_davis.pkl", "tapvid_kinetics_sampled.pkl"],
}


# Main function to execute the download process
def save_data(save_dir, datasets):
    os.makedirs(save_dir, exist_ok=True)

    for data in datasets:
        logger.info(f"Loading {data}.")
        gcloud_path = os.path.join(base_url, data)
        utils.download_from_gcloud(gcloud_path, os.path.join(save_dir, data))


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Specify the path where videos will be downloaded",
    )

    # Add arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["davis", "kinetics", "all"],
        default="all",  # Default path if not specified
        help="Specify the dataset to be downloaded",
    )

    args = parser.parse_args()

    save_data(args.save_dir, available_datasets[args.dataset])
