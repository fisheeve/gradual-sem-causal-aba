
import sys
sys.path.insert(0, 'ArgCausalDisco/')
sys.path.insert(0, 'notears/')

from pathlib import Path
import pytest

from ArgCausalDisco.utils.helpers import random_stability
from ArgCausalDisco.abapc import ABAPC

from src.abapc import get_arrow_sets
from src.utils.bn_utils import get_dataset


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_causalaba_equal_to_ABASP(seed):
    data, _ = get_dataset('cancer', seed=seed)

    res_path = Path('./test_results')
    res_path.mkdir(exist_ok=True)
    random_stability(seed)
    models, _ = ABAPC(data=data,
                    alpha=0.01,
                    indep_test='fisherz',
                    scenario='test_cancer_5_nodes',
                    out_mode="optN")
    
    abasp_models, _, _, _ = get_arrow_sets(data, seed=seed)
    abasp_models = {frozenset(model) for model in abasp_models}

    print(seed)
    print(models)
    print(abasp_models)
    assert abasp_models == models
    print("ABASP and ABAPC models are equal, test passed successfully!")


if __name__ == "__main__":
    test_causalaba_equal_to_ABASP(seed=42)
