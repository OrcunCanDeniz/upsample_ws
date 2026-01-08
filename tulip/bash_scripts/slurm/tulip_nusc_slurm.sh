#!/bin/bash -l 
#SBATCH --export=ALL
#SBATCH --job-name=TULIP_train               # your job name
#SBATCH --nodes=1                          # 1 node
#SBATCH --ntasks-per-node=1                # one srun task per node
#SBATCH --gres=gpu:a100:3               # 8 GPUs on that node
#SBATCH --cpus-per-task=48                  # CPUs for data loading, etc.
#SBATCH --time=24:00:00                    # hh:mm:ss walltime
#SBATCH --partition=a100                 # GPU partition
# (no --output/--error here—handled by srun instead)

# Paths (adjust to your environment)
export SIF_IMAGE="$HOME/conda_upsample_latest.sif"
export WORKSPACE="$HOME/upsample_ws"
export PROJECT="tulip"
export CODE_DIR="${WORKSPACE}/${PROJECT}"
export DATA_DIR="${WORK}/data/nuscenes/"
export CONTAINER_WS="/workspace/${PROJECT}"

# Where to stage outputs/logs
LOCAL_ARTEFACTS_DIR="${HOME}/${PROJECT}_outputs"
NODE_LOCAL_ARTEFACTS_DIR="${TMPDIR}/${PROJECT}_outputs/${SLURM_JOB_ID}"
SLURM_LOGS_DIR="${TMPDIR}/logs"

cp $WORK/data/nuscenes/cmtulip_bevformer_train.tar $TMPDIR/
mkdir -p $TMPDIR/nusc_dataset && tar -xf $TMPDIR/cmtulip_bevformer_train.tar -C $TMPDIR/nusc_dataset

# Go to your code root
cd "${CODE_DIR}"

# Bind code, data—and also the node‐local scratch—into the container
BIND_LIST="${CODE_DIR}:${CONTAINER_WS},${DATA_DIR}:${CONTAINER_WS}/data/nuscenes,${TMPDIR}:${TMPDIR}"

# CA on host, resolve the real file, bind to a fixed path in the container
export HOST_CA_LINK="/etc/pki/tls/certs/ca-bundle.crt"
export HOST_CA_REAL="$(readlink -f "$HOST_CA_LINK")"
export IN_CA="/opt/ssl/cacert.pem"
# BIND_LIST="${BIND_LIST},${HOST_CA_REAL}:${IN_CA}"

# Proxies
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export no_proxy="localhost,127.0.0.1,.nhr.fau.de"

# Point clients to the CA inside the container
export SSL_CERT_FILE="$IN_CA"
export REQUESTS_CA_BUNDLE="$IN_CA"
export CURL_CA_BUNDLE="$IN_CA"
export GIT_SSL_CAINFO="$IN_CA"

# Forward into container
export APPTAINERENV_http_proxy="$http_proxy"
export APPTAINERENV_https_proxy="$https_proxy"
export APPTAINERENV_no_proxy="$no_proxy"
export APPTAINERENV_SSL_CERT_FILE="$SSL_CERT_FILE"
export APPTAINERENV_REQUESTS_CA_BUNDLE="$REQUESTS_CA_BUNDLE"
export APPTAINERENV_CURL_CA_BUNDLE="$CURL_CA_BUNDLE"
export APPTAINERENV_GIT_SSL_CAINFO="$GIT_SSL_CAINFO"
export APPTAINERENV_WANDB_API_KEY="$(< "${HOME}/.wandb_key")"
export APPTAINERENV_CODE_DIR="$CONTAINER_WS"

# Training
apptainer exec --nv \
  --bind "${BIND_LIST}" \
  "${SIF_IMAGE}" \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate py38 && cd "$CODE_DIR" && ./bash_scripts/tulip_upsampling_nusc.sh "${SLURM_GPUS_ON_NODE}"'