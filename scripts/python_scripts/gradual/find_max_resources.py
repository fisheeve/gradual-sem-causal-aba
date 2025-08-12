"""
This script finds the maximum neighbourhood parameter and the maximum conditioning set size parameters.
The priority is:
    1. collider arguments,
    2. neighbourhood size,
    3. conditioning set size.

This means that it first sets the collider arguments to True,
then finds greatest neighbourhood size with c-set size 0,
then finds greatest c-set size with neighbourhood size determined above.
"""
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2, check_memory_usage
from src.utils.resource_utils import MemoryUsageExceededException
import time
import pandas as pd
from logger import logger
from pathlib import Path


RESULTS_DIR = Path("./results/gradual/find_max_resources")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_NODES = [8, 11, 20]  # Number of nodes corresponding to datasets asia, sachs, child

MIN_NEIGHBOURHOOD_SIZE = 3
MIN_C_SET_SIZE = 0


def check_resources_sufficient(use_collider_arguments,
                               neighbourhood_n_nodes,
                               c_set_size):
    start_bsaf_creation = time.time()
    memory_usage_exceeded = False
    try:
        bsaf_builder = BSAFBuilderV2(
            n_nodes=N_NODES,
            include_collider_tree_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            max_conditioning_set_size=c_set_size)  # everything is maximal for given n_nodes, full scale
        bsaf = bsaf_builder.create_bsaf()
        memory_usage = check_memory_usage()
    except MemoryUsageExceededException as e:
        logger.info(f"neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                    f"c_set_size={c_set_size}. Error: {e}")
        memory_usage = check_memory_usage()
        memory_usage_exceeded = True

    elapsed_bsaf_creation = time.time() - start_bsaf_creation
    return memory_usage_exceeded, memory_usage, elapsed_bsaf_creation


def update_and_save_data(data,
                         path,
                         n_nodes,
                         use_collider_arguments,
                         neighbourhood_n_nodes,
                         c_set_size,
                         memory_usage_exceeded,
                         memory_usage,
                         elapsed_bsaf_creation):
    data = pd.concat([data, pd.DataFrame({
        'n_nodes': [n_nodes],
        'use_collider_arguments': [use_collider_arguments],
        'neighbourhood_n_nodes': [neighbourhood_n_nodes],
        'c_set_size': [c_set_size],
        'memory_usage_exceeded': [memory_usage_exceeded],
        'memory_usage_pct': [memory_usage],
        'bsaf_creation_elapsed_time': [elapsed_bsaf_creation]
    })], ignore_index=True)
    data.to_csv(path, index=False)
    return data


def main():
    data = pd.DataFrame(columns=['n_nodes',
                                 'use_collider_arguments',
                                 'neighbourhood_n_nodes',
                                 'c_set_size',
                                 'memory_usage_exceeded',
                                 'memory_usage_pct',
                                 'bsaf_creation_elapsed_time'])
    path = RESULTS_DIR / 'run_metadata.csv'
    for n_nodes in N_NODES:
        logger.info(f"Processing case with {n_nodes} nodes")
        memory_usage_exceeded = False
        neighbourhood_n_nodes = MIN_NEIGHBOURHOOD_SIZE
        c_set_size = MIN_C_SET_SIZE
        use_collider_arguments = True

        # Try with collider arguments first equal to True
        memory_usage_exceeded, memory_usage, elapsed_bsaf_creation = check_resources_sufficient(
            use_collider_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            c_set_size=c_set_size)
        data = update_and_save_data(data, path,
                                    n_nodes, use_collider_arguments,
                                    neighbourhood_n_nodes, c_set_size,
                                    memory_usage_exceeded, memory_usage,
                                    elapsed_bsaf_creation)
        if memory_usage_exceeded:
            # try with less resources by setting collider arguments to False
            use_collider_arguments = False
        else:
            # Try with more resources by increasing neighbourhood size
            neighbourhood_n_nodes += 1

        while True:
            memory_usage_exceeded, memory_usage, elapsed_bsaf_creation = check_resources_sufficient(
                use_collider_arguments=use_collider_arguments,
                neighbourhood_n_nodes=neighbourhood_n_nodes,
                c_set_size=c_set_size)
            data = update_and_save_data(data, path,
                                        n_nodes, use_collider_arguments,
                                        neighbourhood_n_nodes, c_set_size,
                                        memory_usage_exceeded, memory_usage,
                                        elapsed_bsaf_creation)
            if memory_usage_exceeded:
                # If memory usage exceeded, break and go one step back
                logger.info(f"Memory usage exceeded for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                            f"c_set_size={c_set_size}.")
                neighbourhood_n_nodes -= 1
                break
            else:
                # If memory usage is fine, try with more resources by increasing neighbourhood size
                logger.info(f"Memory usage is fine for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                            f"c_set_size={c_set_size}.")
                neighbourhood_n_nodes += 1

        # If neighbourhood size is less than minimum, then no resource is enough
        if neighbourhood_n_nodes < MIN_NEIGHBOURHOOD_SIZE:
            logger.info(f"Memory usage exceeded for minimum resource parameters for {n_nodes} nodes.")
            continue

        # Now try with the maximum neighbourhood size found, but increase c_set_size
        c_set_size += 1
        while True:
            memory_usage_exceeded, memory_usage, elapsed_bsaf_creation = check_resources_sufficient(
                use_collider_arguments=use_collider_arguments,
                neighbourhood_n_nodes=neighbourhood_n_nodes,
                c_set_size=c_set_size)
            data = update_and_save_data(data, path,
                                        n_nodes, use_collider_arguments,
                                        neighbourhood_n_nodes, c_set_size,
                                        memory_usage_exceeded, memory_usage,
                                        elapsed_bsaf_creation)
            if memory_usage_exceeded:
                # If memory usage exceeded, break and go one step back
                logger.info(f"Memory usage exceeded for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                            f"c_set_size={c_set_size}.")
                c_set_size -= 1
                break
            else:
                # If memory usage is fine, try with more resources by increasing c_set_size
                logger.info(f"Memory usage is fine for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                            f"c_set_size={c_set_size}.")
                c_set_size += 1

        logger.info(f"Maximum resources found for {n_nodes} nodes: "
                    f"neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                    f"c_set_size={c_set_size}, "
                    f"use_collider_arguments={use_collider_arguments}.")

        logger.info(f"Completed processing for {n_nodes} nodes\n")


if __name__ == "__main__":
    main()
