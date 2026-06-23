# run_experiments.py
# Complete experiment runner for Impact-VCube

import math
import random
import numpy as np
import pandas as pd

def hamming(a, b):
    return bin(a ^ b).count("1")

def compute_phi(weights, mapping, root=0):
    return sum(weights[p] * hamming(mapping[p], root)
               for p in range(len(weights)))

def impact_threshold_metrics(weights, mapping, root=0, threshold=0.5):
    entries = [(hamming(mapping[p], root), weights[p])
               for p in range(len(weights))]
    entries.sort(key=lambda x: x[0])

    cumulative = 0.0
    nodes = 0
    layer = 0

    for dist, impact in entries:
        cumulative += impact
        nodes += 1
        layer = dist
        if cumulative >= threshold:
            break

    return layer, nodes

def hypercube_neighbors(v, d):
    return [v ^ (1 << i) for i in range(d)]

def random_weights(n):
    w = np.random.rand(n)
    return w / np.sum(w)

def zipf_weights(n, alpha=1.0):
    ranks = np.arange(1, n + 1)
    w = 1.0 / (ranks ** alpha)
    w /= np.sum(w)
    random.shuffle(w)
    return w

def concentrated_weights(n):
    w = np.zeros(n)
    top_k = max(1, int(0.1 * n))
    w[:top_k] = 0.6 / top_k
    w[top_k:] = 0.4 / (n - top_k)
    random.shuffle(w)
    return w

def random_mapping(n):
    ids = list(range(n))
    random.shuffle(ids)
    return {p: ids[p] for p in range(n)}

def optimal_mapping(weights, root=0):
    n = len(weights)
    processes = sorted(range(n), key=lambda p: weights[p], reverse=True)
    positions = sorted(range(n), key=lambda pos: hamming(pos, root))
    return {p: pos for p, pos in zip(processes, positions)}

def local_ordering(weights, root=0):
    n = len(weights)
    d = int(math.log2(n))

    mapping = random_mapping(n)
    initial_mapping = mapping.copy()

    owner = {mapping[p]: p for p in range(n)}
    phi_history = [compute_phi(weights, mapping, root)]

    rounds = 0
    total_swaps = 0

    while True:
        candidates = []

        for pos in range(n):
            proc_a = owner[pos]
            impact_a = weights[proc_a]
            dist_a = hamming(pos, root)

            for neigh in hypercube_neighbors(pos, d):
                if neigh <= pos:
                    continue

                proc_b = owner[neigh]
                impact_b = weights[proc_b]
                dist_b = hamming(neigh, root)

                if ((impact_a - impact_b) * (dist_a - dist_b)) > 0:
                    candidates.append((pos, neigh))

        if not candidates:
            break

        selected = []
        used = set()
        random.shuffle(candidates)

        for a, b in candidates:
            if a not in used and b not in used:
                selected.append((a, b))
                used.add(a)
                used.add(b)

        for a, b in selected:
            pa = owner[a]
            pb = owner[b]

            mapping[pa] = b
            mapping[pb] = a

            owner[a] = pb
            owner[b] = pa

        rounds += 1
        total_swaps += len(selected)
        phi_history.append(compute_phi(weights, mapping, root))

    return {
        "mapping": mapping,
        "initial_mapping": initial_mapping,
        "rounds": rounds,
        "swaps": total_swaps,
        "phi_history": phi_history
    }

def evaluate_size(n, generator, runs=100):

    data = {
        "swaps": [], "rounds": [],
        "gain": [], "gap": [],
        "layer_before": [], "layer_after": [],
        "node_before": [], "node_after": [],
        "layer_gain": [], "node_gain": []
    }

    for _ in range(runs):

        weights = generator(n)
        local = local_ordering(weights)

        phi_initial = local["phi_history"][0]
        phi_final = local["phi_history"][-1]

        optimal = optimal_mapping(weights)
        phi_opt = compute_phi(weights, optimal)

        data["gain"].append(
            ((phi_initial - phi_final) / phi_initial) * 100
        )

        data["gap"].append(
            ((phi_final - phi_opt) / phi_opt) * 100
        )

        l0, n0 = impact_threshold_metrics(
            weights,
            local["initial_mapping"]
        )

        l1, n1 = impact_threshold_metrics(
            weights,
            local["mapping"]
        )

        data["layer_before"].append(l0)
        data["layer_after"].append(l1)

        data["node_before"].append(n0)
        data["node_after"].append(n1)

        data["layer_gain"].append(
            ((l0 - l1) / max(l0, 1)) * 100
        )

        data["node_gain"].append(
            ((n0 - n1) / n0) * 100
        )

        data["swaps"].append(local["swaps"])
        data["rounds"].append(local["rounds"])

    def stats(v):
        mean = np.mean(v)
        std = np.std(v, ddof=1)
        ci95 = 1.96 * std / np.sqrt(len(v))
        return mean, std, ci95

    result = {}

    for metric, values in data.items():
        m, s, c = stats(values)
        result[f"{metric}_mean"] = m
        result[f"{metric}_std"] = s
        result[f"{metric}_ci95"] = c

    return result

def run_experiment():

    sizes = [8,16,32,64,128,256,512,1024]

    distributions = {
        "Random": random_weights,
        "Zipf": zipf_weights,
        "Concentrated": concentrated_weights
    }

    rows = []

    for dist_name, generator in distributions.items():

        print("\\n" + "="*70)
        print(dist_name)
        print("="*70)

        for n in sizes:

            r = evaluate_size(n, generator, runs=100)

            rows.append({
                "distribution": dist_name,
                "n": n,
                **r
            })

            print(
                f"n={n:4d} "
                f"gain={r['gain_mean']:.2f}% "
                f"gap={r['gap_mean']:.2f}% "
                f"layers {r['layer_before_mean']:.2f}->{r['layer_after_mean']:.2f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv("impact_vcube_results.csv", index=False)

    print("\\nSaved: impact_vcube_results.csv")

if __name__ == "__main__":
    run_experiment()
