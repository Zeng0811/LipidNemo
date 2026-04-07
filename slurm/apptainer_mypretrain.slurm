#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2     # n gpus per machine <required>
#SBATCH --cpus-per-gpu=4
#SBATCH -w gpu05
#SBATCH --time=1000:00:00
#SBATCH --job-name=LNPnemo_pt

set -x

while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--prop)
            echo 'Overwriting values from $2.'
            PROPERTY_FILES=$2
            shift
            shift
            ;;
        *)
            echo 'Invalid input'
            exit 1
            ;;
    esac
done

# All variables with default values must be defined in this section
#=========================
STORAGE_ROOT=""
export WANDB_API_KEY="YOUR WANDB API KEY"
MICRO_BATCH_SIZE=32 # Please check GPU mem size. 256 is recommended for A100 with 80 GB mem.
VAL_CHECK_INTERVAL=200
JOB_TYPE='nemo-chem'
EXP_NAME_PREFIX='nemo_chem'
#=========================

set -e
# Any value that needs to be overwritten should be defined in PROPERTY_FILES
if [ ! -z "${PROPERTY_FILES}" ];
then
    IFS=',' read -ra FILES <<< ${PROPERTY_FILES}
    for PROPERTY_FILE in "${FILES[@]}"; do
        source ${PROPERTY_FILE}
    done
fi

if [ -z "${STORAGE_ROOT}" ];
then
    echo "STORAGE_ROOT is invaild. STORAGE_ROOT=${STORAGE_ROOT}. Please check the properties file."
    exit 1
fi

EXP_NAME=${EXP_NAME_PREFIX}_node_${SLURM_JOB_NUM_NODES}_inv${VAL_CHECK_INTERVAL}
DATA_PATH="${STORAGE_ROOT}/megamolbart_data"
RESULT_PATH="${STORAGE_ROOT}/bionemo_results/${EXP_NAME}"
MODEL_PATH="${STORAGE_ROOT}/models"
MOUNTS="$DATA_PATH:/data,$RESULT_PATH:/workspace/bionemo/results,$MODEL_PATH:/workspace/bionemo/models"
mkdir -p ${RESULT_PATH}

# This configuration assumes TP and PP is 1.
GLOBAL_BATCH_SIZE=$(expr ${MICRO_BATCH_SIZE} \* ${SLURM_JOB_NUM_NODES} \* ${SLURM_NTASKS_PER_NODE})

# NeMo and BioNeMo code is picked from the container. To use code from a shared
# folder instead, please NEMO_CODE and BIONEMO_CODE in the properties file.
if [ ! -z "${NEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${NEMO_CODE}:/opt/nvidia/nemo"
fi

if [ ! -z "${BIONEMO_CODE}" ];
then
    MOUNTS="${MOUNTS},${BIONEMO_CODE}/my_pretrain.py:/workspace/bionemo/examples/molecule/megamolbart/my_pretrain.py"
    MOUNTS="${MOUNTS},${BIONEMO_CODE}/finetune_config.yaml:/workspace/bionemo/examples/molecule/megamolbart/conf/finetune_config.yaml"
    MOUNTS="${MOUNTS},${BIONEMO_CODE}/pretrain_base.yaml:/workspace/bionemo/examples/molecule/megamolbart/conf/my_pretrain_base.yaml"
    MOUNTS="${MOUNTS},${BIONEMO_CODE}/my_pretrain_xsmall_span_aug.yaml:/workspace/bionemo/examples/molecule/megamolbart/conf/pretrain_xsmall_span_aug.yaml"
fi

module load apptainer

set -x
srun \
    --output slurm-%j-%n.out \
    --error error-%j-%n.out \
    apptainer exec --nv \
    --bind ${MOUNTS} \
    $pwd/Bionemo/bionemo.sif \
    python /workspace/bionemo/examples/molecule/megamolbart/my_pretrain.py \
        --config-path=/workspace/bionemo/examples/molecule/megamolbart/conf \
        --config-name=my_pretrain_base \
        ++exp_manager.wandb_logger_kwargs.job_type="${JOB_TYPE}" \
        ++exp_manager.wandb_logger_kwargs.name="${EXP_NAME}"\
        ++trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
        ++trainer.devices=${SLURM_GPUS_PER_NODE} \
        ++trainer.accelerator=gpu \
        ++trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
        ++trainer.max_steps=2000000 \
        model.micro_batch_size=${MICRO_BATCH_SIZE} \
        ++exp_manager.resume_if_exists=True \
        restore_from_path=null
set +x
