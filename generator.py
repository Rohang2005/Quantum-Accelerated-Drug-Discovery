import random
BASE_MOLECULES = [
    "C-C-O", "C-O-H", "C=C-O",
    "C-C-N", "C-N-H", "C=C-N",
    "C-O-O", "C-C-C", "C-N-N"
]

def generate_molecules(n=50, bias_tokens=None):
    molecules = []

    for _ in range(n):
        mol = random.choice(BASE_MOLECULES)

        if bias_tokens:
            for token in bias_tokens:
                if random.random() < 0.5:
                    mol += f"-{token}"

        molecules.append(mol)

    return molecules
