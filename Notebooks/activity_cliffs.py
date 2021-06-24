
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
import numpy as np

def cliffs_finder(smiles, y, activity_thr=1, similarity_thr=0.90):

    fps = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(s)) for s in smiles]

    iscliff = []
    index_cliff = []
    max_similarities = []

    for i in range(0, len(fps)):
        sim = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:])    # computes all similarities
        sim[i] = -999   # artificially sets self-similarity to -999
        max_sim = np.max(sim)
        max_similarities.append(max_sim)
        if max_sim > similarity_thr:
            neigh_index = sim.index(max(sim))
            delta = np.abs(y[i] - y[neigh_index])
            if delta > activity_thr:
                iscliff.append(True)
                index_cliff.append(neigh_index)
            else:
                iscliff.append(False)
                index_cliff.append('nan')
        else:
            iscliff.append(False)
            index_cliff.append('nan')

    return iscliff, index_cliff, max_similarities


def cliffs_finder_test(smiles_train, smiles_test, y_train, y_test, activity_thr=1, similarity_thr=0.90):

    fps_train = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(s)) for s in smiles_train]
    fps_test = [FingerprintMols.FingerprintMol(Chem.MolFromSmiles(s)) for s in smiles_test]

    iscliff = []
    index_cliff = []
    max_similarities = []

    for i in range(0, len(fps_test)):
        sim = DataStructs.BulkTanimotoSimilarity(fps_test[i], fps_train[:])    # computes all similarities
        max_sim = np.max(sim)

        max_similarities.append(max_sim)
        if max_sim > similarity_thr:
            neigh_index = sim.index(max(sim))
            if i < len(y_test) and neigh_index < len(y_train):
                delta = np.abs(y_test[i] - y_train[neigh_index])
                if delta > activity_thr:
                    iscliff.append(True)
                    index_cliff.append(neigh_index)
                else:
                    iscliff.append(False)
                    index_cliff.append('nan')
        else:
            iscliff.append(False)
            index_cliff.append('nan')

    return iscliff, index_cliff, max_similarities