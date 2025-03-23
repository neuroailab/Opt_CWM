DATASET=$1 
EVALUATION=$2 

tapvid_datasets=("davis" "kinetics")

if [[ ! " ${tapvid_datasets[@]} " =~ " ${DATASET} " ]]; then
  echo "ERROR: Unknown TAP-Vid Dataset: $DATASET" 
  exit 1
fi 

# set frame_delta >= 0 for CFG eval (use 5 in paper)
if [ "$EVALUATION" = "cfg" ]; then
    FRAME_DELTA=5
elif [ "$EVALUATION" = "first" ]; then
    FRAME_DELTA=-1
else
    echo "ERROR: Unknown Evaluation Mode: $EVALUATION"
    exit 1
fi

NPROC="${NPROC:=8}"  # set as env variable if needed

shift 2
flags=()
for arg in "$@"; do
    if [[ $arg == --* ]]; then
        flags+=("$arg")
    else 
        echo "WARNING: Ignoring unknown flag format: \"$arg\""
    fi
done

set -xeo pipefail 

torchrun --nproc_per_node=$NPROC \
 tapvid_eval.py \
 --yaml=configs/raft_eval_cfg.yaml \
 --data_args.tapvid.dataset=$DATASET \
 --data_args.tapvid.frame_delta=$FRAME_DELTA \
 "${flags[@]}"
