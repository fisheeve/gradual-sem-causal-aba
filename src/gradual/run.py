from pathlib import Path
from dataclasses import dataclass

from src.gradual.model_wrapper import ModelWrapper, ModelEnum
from src.causal_aba.core_factory import CoreABASPSolverFactory

from GradualABA.semantics.modular.SetProductAggregation import SetProductAggregation
from GradualABA.BSAF import BSAF

from logger import logger

@dataclass
class GradualCausalABAOutput:
    arrow_strengths: dict
    indep_strengths: dict
    greedy_graph: dict
    graph_data: dict
    bsaf: BSAF


def run(X_s,
        factory: CoreABASPSolverFactory,
        model_name: ModelEnum = ModelEnum.DF_QUAD,
        set_aggregation=SetProductAggregation(),
        aggregation=None,
        influence=None,
        conservativeness: float = 1.0):
    """
    Run the gradual causal ABA solver with the given parameters.

    :param X_s: Input data for the model.
    :param factory: Factory to create the ABAF solver.
    :param model_name: Name of the model to use.
    :param set_aggregation: Set aggregation method to use.
    :param aggregation: Aggregation method to use.
    :param influence: Influence method to use.
    :param conservativeness: Conservativeness factor for the model.

    :return: GradualCausalABAOutput containing arrow strengths, indep strengths, and greedy graph.
    """
    if ((set_aggregation is None
         or aggregation is None
         or influence is None)
            and model_name is None):
        raise ValueError(
            "Please provide either model_name or all three of (set_aggregation, aggregation, influence).")
    logger.info(f"Creating solver with model_name: {model_name}, "
                f"set_aggregation: {set_aggregation}, "
                f"aggregation: {aggregation}, "
                f"influence: {influence}, "
                f"conservativeness: {conservativeness}, ")

    abaf_builder = factory.create_solver()
    abaf = abaf_builder.get_abaf()
    bsaf = abaf.to_bsaf(weight_agg=set_aggregation)

    logger.info("solving BSAF with GradualCausalABA")
    model_wrapper = ModelWrapper(
        bsaf=bsaf,
        n_nodes=X_s.shape[1],
        model_name=model_name,
        set_aggregation=set_aggregation,
        aggregation=aggregation,
        influence=influence,
        conservativeness=conservativeness)
    graph_data = model_wrapper.solve(iterations=10,
                                     verbose=True)

    return GradualCausalABAOutput(
        arrow_strengths=model_wrapper.get_arrow_strengths(),
        indep_strengths=model_wrapper.get_indep_strengths(),
        greedy_graph=model_wrapper.build_greedy_graph(),
        graph_data=graph_data,
        bsaf=bsaf,
    )
