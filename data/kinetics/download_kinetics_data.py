import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Base URL for downloading files
base_url = "https://storage.googleapis.com/kinetics-400-public/Kinetics400/k400/train"


# Function to download a file using curl
def download_video(video_filename, destination_folder):
    video_url = f"{base_url}/{video_filename}"
    destination_path = os.path.join(destination_folder, video_filename)

    # Use subprocess to execute the curl command
    subprocess.run(["curl", video_url, "-o", destination_path, "-s"], check=True)


# Function to download videos in parallel using ThreadPoolExecutor
def download_videos_multithreaded(video_filenames, destination_folder, max_workers=10):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Initialize tqdm progress bar
    with tqdm(total=len(video_filenames), desc="Downloading videos", unit="file") as pbar:
        # Define a wrapper function to update the progress bar after each download
        def download_with_progress(video_filename):
            download_video(video_filename, destination_folder)
            pbar.update(1)  # Increment the progress bar by 1 for each file downloaded

        # Use ThreadPoolExecutor to download files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(download_with_progress, video_filenames)


# Function to load video filenames from a text file
def load_video_filenames(file_path):
    with open(file_path, "r") as file:
        # Read each line and strip newlines or spaces
        video_filenames = [line.strip() for line in file.readlines()]
    return video_filenames


# Main function to execute the download process
def main(txt_file_path, destination_folder, max_workers=10):
    # Load video filenames from the specified text file
    video_filenames = load_video_filenames(txt_file_path)

    # Start downloading videos with the specified destination folder
    download_videos_multithreaded(video_filenames, destination_folder, max_workers)


if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Download Kinetics videos with multithreading")

    # Add arguments
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        default="/path/to/destination/folder",  # Default path if not specified
        help="Specify the path where videos will be downloaded",
    )

    parser.add_argument(
        "--txt_file",
        type=str,
        default="cwm/data/scripts/kinetics_video_names.txt",  # Default text file
        help="Path to the text file containing the list of video filenames",
    )

    parser.add_argument(
        "--max_workers", type=int, default=96, help="Maximum number of threads to use for parallel downloads"
    )

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.txt_file, args.save_path, args.max_workers)
