flags=()
for arg in "$@"; do
    if [[ $arg == --* ]]; then
        flags+=("$arg")
    else 
        echo "WARNING:Ignoring unknown flag format: \"$arg\""
    fi
done

set -xeo pipefail 

python demo.py --yaml=configs/eval_cfg.yaml \
 --model_args.flow_predictor.masking_iters=5 \
 --model_args.flow_predictor.zoom_iters=4 \
 "${flags[@]}"