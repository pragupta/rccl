#!/bin/bash

#SEQUENCE_LENGTHS=(50 128 256 512 1024 2048 4096)
#ALL_REDUCE_ALGOS=(1 2 3 4)

SEQUENCE_LENGTHS=(50)
ALL_REDUCE_ALGOS=(1)
HIP_DEV_FORCE_KERNARG=1

for SEQ_LEN in "${SEQUENCE_LENGTHS[@]}"; do
	for ALGO in "${ALL_REDUCE_ALGOS[@]}"; do
		echo "Running sequence length $SEQ_LEN with intra-node all_reduce $ALGO"
		ENABLE_INTRA_NODE_COMM=1 runTracer.sh torchrun --nproc_per_node=2 all_reduce.py --sequence_lengths $SEQ_LEN --all_reduce $ALGO #--tracing
	done
done
