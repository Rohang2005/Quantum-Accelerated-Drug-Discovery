from generator import generate_molecules
from quantscore import quantum_score
from classifier import classify_molecules
from feature_extractor import extract_common_features
from feedback import update_bias
from explain import explain_selection

ITERATIONS = 3
MOLECULES_PER_ITER = 50


def run_pipeline(include_explanation=True):
    bias = None
    iteration_logs = []

    for iteration in range(ITERATIONS):
        log_line = f"Iteration {iteration + 1}/{ITERATIONS}"
        iteration_logs.append(log_line)

        molecules = generate_molecules(
            n=MOLECULES_PER_ITER,
            bias_tokens=bias
        )

        scores = [quantum_score(mol) for mol in molecules]

        good, bad = classify_molecules(molecules, scores)

        iteration_logs.append(f"  Good molecules: {len(good)} | Bad molecules: {len(bad)}")

        features = extract_common_features(good)

        iteration_logs.append(f"  Learned favorable features: {features}")

        bias = update_bias(features)

    final_candidates = sorted(good, key=lambda x: x[1])[:5]

    explanation = None
    if include_explanation:
        explanation = explain_selection(
            top_molecules=final_candidates,
            learned_features=bias,
            iterations=ITERATIONS
        )

    return final_candidates, explanation, iteration_logs


if __name__ == "__main__":
    final_candidates, explanation, iteration_logs = run_pipeline(include_explanation=True)

    print("\n=== ITERATION PROGRESS ===")
    for line in iteration_logs:
        print(line)

    print("\n=== FINAL CANDIDATES ===")
    for mol, score in final_candidates:
        print(mol, "→ Quantum Score:", score)

    print("\n=== LLM EXPLANATION ===")
    print(explanation)