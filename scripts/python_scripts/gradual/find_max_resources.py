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
from src.gradual.scalable.bsaf_builder_v2 import BSAFBuilderV2, check_memory_usage_gb
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


def check_resources_sufficient(data,
                               path,
                               n_nodes,
                               use_collider_arguments,
                               neighbourhood_n_nodes,
                               c_set_size):
    start_bsaf_creation = time.time()
    memory_usage_exceeded = False
    try:
        bsaf_builder = BSAFBuilderV2(
            n_nodes=n_nodes,
            include_collider_tree_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            max_conditioning_set_size=c_set_size)  # everything is maximal for given n_nodes, full scale
        bsaf = bsaf_builder.create_bsaf()
        memory_usage = check_memory_usage_gb()
    except MemoryUsageExceededException as e:
        logger.info(f"neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                    f"c_set_size={c_set_size}. Error: {e}")
        memory_usage = check_memory_usage_gb()
        memory_usage_exceeded = True

    elapsed_bsaf_creation = time.time() - start_bsaf_creation

    data = update_and_save_data(data,
                                path,
                                n_nodes,
                                use_collider_arguments,
                                neighbourhood_n_nodes,
                                c_set_size,
                                memory_usage_exceeded,
                                memory_usage,
                                elapsed_bsaf_creation)

    return data, memory_usage_exceeded, memory_usage, elapsed_bsaf_creation


def find_max_c_set_size(n_nodes,
                        min_c_set_size,
                        use_collider_arguments,
                        neighbourhood_n_nodes,
                        path,
                        data):
    """
    Finds the maximum c-set size for given n_nodes, use_collider_arguments and neighbourhood_n_nodes.
    It starts with c_set_size = min_c_set_size and increases it until memory usage exceeds the limit.
    """
    c_set_size = min_c_set_size
    while True:
        if c_set_size > n_nodes - 2:
            logger.info(f"Conditioning set size {c_set_size} exceeds number of nodes {n_nodes}. "
                        f"Breaking loop.")
            c_set_size = n_nodes - 2
            break

        data, memory_usage_exceeded, memory_usage, elapsed_bsaf_creation = check_resources_sufficient(
            data=data,
            path=path,
            n_nodes=n_nodes,
            use_collider_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            c_set_size=c_set_size)

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
    return c_set_size, data


def find_max_neighbourhood_size(n_nodes,
                                min_neighbourhood_n_nodes,
                                use_collider_arguments,
                                c_set_size,
                                path,
                                data):
    """ Finds the maximum neighbourhood size for given n_nodes, use_collider_arguments and c_set_size.
    It starts with neighbourhood_n_nodes = min_neighbourhood_n_nodes and increases it
    until memory usage exceeds the limit.
    """
    neighbourhood_n_nodes = min_neighbourhood_n_nodes
    while True:
        if neighbourhood_n_nodes > n_nodes:
            logger.info(f"Neighbourhood size {neighbourhood_n_nodes} exceeds number of nodes {n_nodes}. "
                        f"Breaking loop.")
            neighbourhood_n_nodes = n_nodes
            break

        data, memory_usage_exceeded, memory_usage, elapsed_bsaf_creation = check_resources_sufficient(
            data=data,
            path=path,
            n_nodes=n_nodes,
            use_collider_arguments=use_collider_arguments,
            neighbourhood_n_nodes=neighbourhood_n_nodes,
            c_set_size=c_set_size)
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
    return neighbourhood_n_nodes, data


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
        for use_collider_arguments in [True, False]:
            # Find maximum neighbourhood size for the current collider arguments
            max_neighbourhood_n_nodes, data = find_max_neighbourhood_size(
                n_nodes=n_nodes,
                min_neighbourhood_n_nodes=MIN_NEIGHBOURHOOD_SIZE,
                use_collider_arguments=use_collider_arguments,
                c_set_size=MIN_C_SET_SIZE,
                path=path,
                data=data)
            logger.info(f"Maximum neighbourhood_n_nodes found: {max_neighbourhood_n_nodes} for "
                        f"use_collider_arguments={use_collider_arguments}.")

            # If neighbourhood size is less than minimum, then no resource is enough
            if max_neighbourhood_n_nodes < MIN_NEIGHBOURHOOD_SIZE:
                logger.info(f"Memory usage exceeded for minimum resource parameters for {n_nodes} nodes.")
                continue
            else:
                # Find maximum c_set_size for each neighbourhood size
                for neighbourhood_n_nodes in range(MIN_NEIGHBOURHOOD_SIZE, max_neighbourhood_n_nodes + 1):
                    logger.info(f"Finding maximum c_set_size for n_nodes={n_nodes}, "
                                f"use_collider_arguments={use_collider_arguments}, "
                                f"neighbourhood_n_nodes={neighbourhood_n_nodes}.")
                    max_c_set_size, data = find_max_c_set_size(
                        n_nodes=n_nodes,
                        min_c_set_size=MIN_C_SET_SIZE + 1,  # Start from 1 to avoid c_set_size = 0 which has already been checked
                        use_collider_arguments=use_collider_arguments,
                        neighbourhood_n_nodes=neighbourhood_n_nodes,
                        path=path,
                        data=data)
                    logger.info(f"Maximum c_set_size found: {max_c_set_size} for neighbourhood_n_nodes={neighbourhood_n_nodes}, "
                                f"use_collider_arguments={use_collider_arguments}.")

        logger.info(f"Completed processing for {n_nodes} nodes\n")


if __name__ == "__main__":
    main()
