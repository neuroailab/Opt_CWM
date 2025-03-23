NPROC="${NPROC:=8}"  # set as env variable if needed
flags=()
for arg in "$@"; do
    if [[ $arg == --* ]]; then
        flags+=("$arg")
    else 
        echo "WARNING:Ignoring unknown flag format: \"$arg\""
    fi
done

set -xeo pipefail 

torchrun --nproc_per_node=$NPROC \
 train.py \
 --yaml=configs/train_cfg.yaml \
 "${flags[@]}"