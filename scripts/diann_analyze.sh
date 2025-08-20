#!/bin/bash
image_path="/scratch/group/jbardlab/containers/diann_docker:v0.1.sif"
data_path="/scratch/user/jbard/Datasets/Pan_2024"

singularity exec \
    "${image_path}"  \
    /bin/bash -c "/diann-2.2.0/diann-linux \
    -h"