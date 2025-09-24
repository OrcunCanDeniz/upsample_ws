export SIF_IMAGE="$HOME/conda_upsample_latest.sif"
export WORKSPACE="$HOME/upsample_ws"
export PROJECT="tulip"
export CODE_DIR="$WORKSPACE/$PROJECT"
export DATA_DIR="$WORK/data/nuscenes"
export CONTAINER_WS="/workspace/$PROJECT"

export WANDB_API_KEY="$(< $HOME/.wandb_key)"
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"

BIND_LIST="${CODE_DIR}:${CONTAINER_WS},${DATA_DIR}:${CONTAINER_WS}/data"
echo $BIND_LIST 

apptainer shell --nv \
    --bind "$BIND_LIST" \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env http_proxy="$http_proxy" \
    --env https_proxy="$https_proxy" \
    "$SIF_IMAGE"
