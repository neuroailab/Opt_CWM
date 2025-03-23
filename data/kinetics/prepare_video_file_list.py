import argparse
import glob
import os

# Initialize argparse
parser = argparse.ArgumentParser("prepare video file list", add_help=False)

# Argument for the folder containing the .mp4 videos
parser.add_argument("--videos_path", type=str, required=True, help="path to the folder with downloaded videos")

# Argument for the output file path where the .txt file will be saved
parser.add_argument("--output_path", type=str, required=True, help="path to save the output text file")

args = parser.parse_args()


def prepare_video_file_list(videos_path, output_path):
    # Get the list of video files in the specified folder
    video_files = glob.glob(os.path.join(args.videos_path, "*.mp4"))

    # Open the output file to write the list of video files
    with open(args.output_path, "w") as file_save:
        for ct, filename in enumerate(video_files):
            # Write the video filename followed by ' 0' (assumed label)
            file_save.write(filename + " 0")

            # Add a newline after each file, except for the last one
            if ct != len(video_files) - 1:
                file_save.write("\n")

    print(f"Video file list saved to {args.output_path}")


if __name__ == "__main__":
    prepare_video_file_list(args.videos_path, args.output_path)
