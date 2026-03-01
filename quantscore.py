from quantumeval import run_vqe
def quantum_score(molecule):
    return run_vqe()
cache = {}

def quantum_score(molecule):
    if molecule not in cache:
        cache[molecule] = run_vqe()
    return cache[molecule]