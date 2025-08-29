from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
import numpy as np


class MoleculeEvaluator:
    def __init__(self, gen_smiles, ref_smiles, radius=2, n_bits=2048):
        self.radius = radius
        self.n_bits = n_bits

        # Deduplicate originals
        gen_smiles = list(set(gen_smiles))
        ref_smiles = list(set(ref_smiles))

        # Save original count (before filtering invalids)
        self.original_gen_count = len(gen_smiles)

        # Keep only valid SMILES
        self.gen_mols, self.gen_smiles = [], []
        for s in gen_smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                self.gen_mols.append(m)
                self.gen_smiles.append(s)

        self.ref_mols, self.ref_smiles = [], []
        for s in ref_smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                self.ref_mols.append(m)
                self.ref_smiles.append(s)

        # Fingerprints
        self.gen_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.n_bits)
            for m in self.gen_mols
        ]
        self.ref_fps = [
            AllChem.GetMorganFingerprintAsBitVect(m, self.radius, nBits=self.n_bits)
            for m in self.ref_mols
        ]

    # Core Metrics

    def internal_diversity(self):
        n = len(self.gen_fps)
        if n < 2:
            return 0.0
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = DataStructs.TanimotoSimilarity(self.gen_fps[i], self.gen_fps[j])
                dists.append(1 - sim)
        return np.mean(dists)

    def nearest_neighbor_similarity(self):
        sims = []
        for fp in self.gen_fps:
            sim = max(DataStructs.BulkTanimotoSimilarity(fp, self.ref_fps)) if self.ref_fps else 0.0
            sims.append(sim)
        return np.mean(sims) if sims else 0.0

    def scaffold_similarity(self):
        def get_scaffolds(mols):
            return {MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols}

        gen_scaff = get_scaffolds(self.gen_mols)
        ref_scaff = get_scaffolds(self.ref_mols)
        inter = len(gen_scaff.intersection(ref_scaff))
        return inter / len(gen_scaff) if gen_scaff else 0.0

    def fragment_similarity(self):
        def get_fragments(mols):
            frags = []
            for m in mols:
                frags.extend(list(BRICS.BRICSDecompose(m)))
            return set(frags)

        gen_frags = get_fragments(self.gen_mols)
        ref_frags = get_fragments(self.ref_mols)
        inter = len(gen_frags.intersection(ref_frags))
        return inter / len(gen_frags) if gen_frags else 0.0

    def novelty(self):
        ref_set = set(self.ref_smiles)
        novel = [s for s in self.gen_smiles if s not in ref_set]
        return len(novel) / self.original_gen_count if self.original_gen_count else 0.0

    def validity(self):
        return len(self.gen_mols) / self.original_gen_count if self.original_gen_count else 0.0

    def unique_at_k(self, k=None):
        smiles = self.gen_smiles[:k] if k else self.gen_smiles
        return len(set(smiles)) / len(smiles) if smiles else 0.0

    def filters(self):
        passed = 0
        for m in self.gen_mols:
            mw = Descriptors.MolWt(m)
            logp = Descriptors.MolLogP(m)
            hbd = Descriptors.NumHDonors(m)
            hba = Descriptors.NumHAcceptors(m)
            if mw < 500 and logp < 5 and hbd <= 5 and hba <= 10:
                passed += 1
        return passed / len(self.gen_mols) if self.gen_mols else 0.0

    def evaluate_all(self, unique_k=None):
        return {
            "InternalDiversity": self.internal_diversity(),
            "NearestNeighborSimilarity": self.nearest_neighbor_similarity(),
            "ScaffoldSimilarity": self.scaffold_similarity(),
            "FragmentSimilarity": self.fragment_similarity(),
            "Novelty": self.novelty(),
            "Validity": self.validity(),
            "Unique@k": self.unique_at_k(unique_k),
            "FiltersPass": self.filters()
        }
