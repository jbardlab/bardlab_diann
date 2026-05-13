#!/bin/bash

# Configuration
IMAGE_NAME="ghcr.io/jbardlab/diann_docker:v2.5.1"
ANALYZE_SCRIPT="/scratch/user/jbard/repos/bardlab_diann/scripts/run_diann_template.sh"

# Pull image if it doesn't already exist
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Pulling Docker image: ${IMAGE_NAME}"
    docker pull "${IMAGE_NAME}"
else
    echo "Docker image already exists locally: ${IMAGE_NAME}"
fi

# Run the container
docker run --rm \
    -v "$(dirname "${ANALYZE_SCRIPT}"):/scripts" \
    "${IMAGE_NAME}" \
    /bin/bash -c "bash /scripts/$(basename "${ANALYZE_SCRIPT}")"