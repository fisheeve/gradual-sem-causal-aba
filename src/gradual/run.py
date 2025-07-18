from dataclasses import dataclass
from typing import List

from src.gradual.model_wrapper import ModelWrapper, ModelEnum
from src.utils.enums import Fact
from src.causal_aba.core_factory import CoreABASPSolverFactory

from GradualABA.semantics.modular.SetProductAggregation import SetProductAggregation
from GradualABA.BSAF import BSAF
from src.gradual.abaf_opt import ABAFOptimised
from GradualABA.ABAF import ABAF
from typing import Union

from logger import logger


@dataclass
class GradualCausalABAOutput:
    arrow_strengths: dict
    indep_strengths: dict
    greedy_graph: dict
    graph_data: dict


def run_get_bsaf(factory: CoreABASPSolverFactory,
                 facts: List[Fact],
                 set_aggregation=SetProductAggregation(),
                 abaf_class: Union[ABAFOptimised, ABAF] = ABAFOptimised) -> BSAF:
    """    Run the factory to create a BSAF with assums strengths based on the given facts.
    """
    abaf_builder = factory.create_solver(facts=facts)
    abaf = abaf_builder.get_abaf(abaf_class=abaf_class)
    bsaf = abaf.to_bsaf(weight_agg=set_aggregation)

    return bsaf


def run_model(n_nodes: int,
              bsaf: BSAF,
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
    logger.info("solving BSAF with GradualCausalABA")
    model_wrapper = ModelWrapper(
        bsaf=bsaf,
        n_nodes=n_nodes,
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
        graph_data=graph_data
    )


if __name__ == "__main__":
    from src.gradual.extra.abaf_factory_v1 import FactoryV1

    # Example usage
    factory = FactoryV1(n_nodes=5)
    facts = [Fact(node1=1,
                  node2=2,
                  relation=Fact.RelationEnum.dep,
                  score=0.8,
                  S={3, 4}),
             Fact(node1=2,
                  node2=3,
                  relation=Fact.RelationEnum.indep,
                  score=0.5,
                  S={4})]
    bsaf = run_get_bsaf(factory, facts)

    output = run_model(n_nodes=5, bsaf=bsaf)
