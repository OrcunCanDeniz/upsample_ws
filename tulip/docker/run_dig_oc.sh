# Configure X11 forwarding for GUI visualization (e.g., Open3D)
# Docker image and container configuration
# Define the user's home directory using the standard HOME variable
HOME_DIR="$HOME"

# Define the project name


# List of projects to mount into the Docker container
PROJECTS_LIST=("tulip")

# Function to generate Docker volume mount strings for a project
# Arguments:
#   $1 - Project name
# Outputs:
#   A string of volume mount options for Docker
populate_volumes() {
  local project_name="$1"
  local data_path="$HOME_DIR/upsample_ws/$project_name/data"
  local code_path="$HOME_DIR/upsample_ws/$project_name"
  local docker_code_path="/workspace/$project_name"
  local volumes=""

  # 1) Mount the whole project
  volumes="--volume=$code_path:$docker_code_path"

  # 2) If data/ exists with symlinks, handle them specially
  if [ -d "$data_path" ]; then
    # Check if there are any symlinks
    if find "$data_path" -type l -print -quit | grep -q .; then
      # Create a temporary directory to recreate data structure without symlinks
      local temp_data_dir=$(mktemp -d)
      
      # Copy the directory structure (directories only, skip symlinks)
      find "$data_path" -type d ! -path "$data_path" -print0 | while IFS= read -r -d '' dir; do
        local rel="${dir#$data_path/}"
        mkdir -p "$temp_data_dir/$rel"
      done
      
      # Mount this temp directory as the data folder (creates proper directory structure)
      volumes="$volumes --volume=$temp_data_dir:$docker_code_path/data"
      
      # Now mount each symlink target on top
      while IFS= read -r -d '' softlink; do
        local rel="${softlink#$data_path/}"
        local softlink_in_docker="$docker_code_path/data/$rel"
        local target; target="$(readlink -f "$softlink")"
        volumes="$volumes --volume=$target:$softlink_in_docker"
      done < <(find "$data_path" -type l -print0)
    else
      # No symlinks, just mount data normally
      volumes="$volumes --volume=$data_path:$docker_code_path/data"
    fi
  fi

  echo "$volumes"
}

TAG="latest"
BUILD_NAME="orcund/conda_upsample"
IMAGE_NAME="${BUILD_NAME,,}:${TAG,,}"
CONTAINER_NAME="upsample"


# Initialize volumes string
VOLUMES=""

# Populate volumes for each project in the list
for project in "${PROJECTS_LIST[@]}"; do
    volumes=$(populate_volumes "$project")
    VOLUMES="$VOLUMES $volumes"
done

# Remove leading/trailing spaces and collapse multiple spaces into one
VOLUMES_CLEAN=$(echo "$VOLUMES" | tr -s ' ' | sed 's/^ //;s/ $//')

# Display the volumes
echo "Mounting the following directories to the Docker container:"
echo "$VOLUMES_CLEAN" | tr ' ' '\n'

# Check if WANDB_API_KEY is set and not empty
if [ -z "${WANDB_API_KEY:-}" ]; then
    read -rp "Enter your WANDB API key: " WANDB_API_KEY
fi

# Add it to the Docker env string
WANDB_ENV="--env WANDB_API_KEY=${WANDB_API_KEY}"

VISUAL="--env=DISPLAY \
        --env=QT_X11_NO_MITSHM=1 \
        --volume=/tmp/.X11-unix:/tmp/.X11-unix"
xhost +local:docker  # Allow Docker to access the X server

# Launch the Docker container
docker run -d -it --rm \
    -p 8888:8888 \
    $VOLUMES \
    $VISUAL \
    $WANDB_ENV \
    --env NVIDIA_DISABLE_REQUIRE=1 \
    --gpus all \
    --privileged \
    --net=host \
    --ipc=host \
    --shm-size=30G \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"
