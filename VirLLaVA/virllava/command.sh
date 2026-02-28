export PYTHONPATH=/home/xzh/LLaVA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

python virllava/demo.py