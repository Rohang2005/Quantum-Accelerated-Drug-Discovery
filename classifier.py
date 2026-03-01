import numpy as np

def classify_molecules(molecules, scores, percentile=30):
    threshold = np.percentile(scores, percentile)

    good = []
    bad = []

    for mol, score in zip(molecules, scores):
        if score <= threshold:
            good.append((mol, score))
        else:
            bad.append((mol, score))

    return good, bad
