# test_molecule_evaluator.py

import pytest
from rdkit import Chem
from src.eval import MoleculeEvaluator  # assuming your class file is named molecule_evaluator.py


@pytest.fixture
def sample_data():
    gen = ["CCO", "CCN", "c1ccccc1", "invalid_smiles", "CCO"]  # duplicate + invalid
    ref = ["CCO", "CCC", "c1ccccc1O"]
    return gen, ref


@pytest.fixture
def evaluator(sample_data):
    gen, ref = sample_data
    return MoleculeEvaluator(gen, ref)


def test_initialization_removes_duplicates_and_invalid(evaluator):
    # Ensure duplicates removed and invalid excluded
    assert "invalid_smiles" not in evaluator.gen_smiles
    assert len(evaluator.gen_smiles) == len(set(evaluator.gen_smiles))


def test_validity(evaluator):
    # 4 gen_smiles, but 1 invalid → 3 valid
    assert evaluator.validity() == 3 / 4


def test_novelty(evaluator):
    # "CCO" appears in ref → not novel
    # "CCN" and "c1ccccc1" are novel
    expected = 2 / 4
    assert evaluator.novelty() == pytest.approx(expected)


def test_unique_at_k(evaluator):
    # duplicates were removed, so uniqueness should be full
    assert evaluator.unique_at_k() == 1.0
    # top-k smaller list
    assert 0.0 <= evaluator.unique_at_k(k=2) <= 1.0


def test_filters(evaluator):
    # All small simple molecules should pass Lipinski filters
    val = evaluator.filters()
    assert 0.0 <= val <= 1.0
    assert val > 0


def test_internal_diversity(evaluator):
    val = evaluator.internal_diversity()
    assert 0.0 <= val <= 1.0
    # Diversity > 0 because multiple distinct molecules
    assert val > 0


def test_nearest_neighbor_similarity(evaluator):
    val = evaluator.nearest_neighbor_similarity()
    assert 0.0 <= val <= 1.0


def test_scaffold_similarity(evaluator):
    val = evaluator.scaffold_similarity()
    assert 0.0 <= val <= 1.0


def test_fragment_similarity(evaluator):
    val = evaluator.fragment_similarity()
    assert 0.0 <= val <= 1.0


def test_evaluate_all_keys(evaluator):
    results = evaluator.evaluate_all(unique_k=2)
    expected_keys = {
        "InternalDiversity",
        "NearestNeighborSimilarity",
        "ScaffoldSimilarity",
        "FragmentSimilarity",
        "Novelty",
        "Validity",
        "Unique@k",
        "FiltersPass",
    }
    assert set(results.keys()) == expected_keys
    for v in results.values():
        assert isinstance(v, float)
