from collections import Counter

def extract_common_features(good_molecules, top_k=3):
    tokens = []

    for mol, _ in good_molecules:
        tokens.extend(mol.split("-"))

    common = Counter(tokens).most_common(top_k)

    return [feature for feature, _ in common]
