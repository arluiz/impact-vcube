import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# UTILIDADES
# =====================================================

def hamming(a, b):
    return bin(a ^ b).count("1")


def compute_phi(weights, mapping, root=0):
    return sum(
        weights[p] * hamming(mapping[p], root)
        for p in range(len(weights))
    )


# =====================================================
# HIPERCUBO
# =====================================================

def hypercube_neighbors(v, d):

    neighbors = []

    for i in range(d):
        neighbors.append(v ^ (1 << i))

    return neighbors


# =====================================================
# DISTRIBUIÇÕES
# =====================================================

def random_weights(n):

    w = np.random.rand(n)

    return w / np.sum(w)


def zipf_weights(n, alpha=1.0):

    ranks = np.arange(1, n + 1)

    w = 1.0 / (ranks ** alpha)
    
    w = w / np.sum(w)
    random.shuffle(w)

    return w


def concentrated_weights(n):

    w = np.zeros(n)

    top_k = max(1, int(0.1 * n))

    w[:top_k] = 0.6 / top_k

    w[top_k:] = 0.4 / (n - top_k)
    
    random.shuffle(w)

    return w


# =====================================================
# MAPEAMENTOS
# =====================================================

def random_mapping(n):

    ids = list(range(n))

    random.shuffle(ids)

    return {
        p: ids[p]
        for p in range(n)
    }


def optimal_mapping(weights, root=0):

    n = len(weights)

    processes = sorted(
        range(n),
        key=lambda p: weights[p],
        reverse=True
    )

    positions = sorted(
        range(n),
        key=lambda pos: hamming(pos, root)
    )

    return {
        p: pos
        for p, pos in zip(processes, positions)
    }


# =====================================================
# IMPACT-VCUBE LOCAL ORDERING
# =====================================================

def local_ordering(weights, root=0):

    n = len(weights)

    d = int(math.log2(n))

    mapping = random_mapping(n)

    owner = {
        mapping[p]: p
        for p in range(n)
    }

    phi_history = [
        compute_phi(weights, mapping, root)
    ]

    rounds = 0

    total_swaps = 0

    while True:

        candidates = []

        # =========================================
        # Encontrar trocas permitidas
        # =========================================

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

                #
                # maior impacto mais distante
                #

                if (
                    (impact_a - impact_b)
                    *
                    (dist_a - dist_b)
                    > 0
                ):

                    candidates.append(
                        (pos, neigh)
                    )

        if len(candidates) == 0:
            break

        # =========================================
        # Matching maximal
        # =========================================

        selected = []

        used = set()

        random.shuffle(candidates)

        for pos_a, pos_b in candidates:

            if pos_a in used:
                continue

            if pos_b in used:
                continue

            selected.append(
                (pos_a, pos_b)
            )

            used.add(pos_a)
            used.add(pos_b)

        # =========================================
        # Executar round
        # =========================================

        for pos_a, pos_b in selected:

            proc_a = owner[pos_a]
            proc_b = owner[pos_b]

            mapping[proc_a] = pos_b
            mapping[proc_b] = pos_a

            owner[pos_a] = proc_b
            owner[pos_b] = proc_a

        total_swaps += len(selected)

        rounds += 1

        phi_history.append(
            compute_phi(weights, mapping, root)
        )

    return {
        "mapping": mapping,
        "swaps": total_swaps,
        "rounds": rounds,
        "phi_history": phi_history
    }


# =====================================================
# AVALIAÇÃO
# =====================================================

def evaluate_size(
        n,
        generator,
        runs=100):

    swaps = []
    rounds = []
    gains = []
    gaps = []

    for _ in range(runs):

        weights = generator(n)
        
        print(weights)

        local = local_ordering(weights)

        phi_initial = local["phi_history"][0]
        phi_local = local["phi_history"][-1]

        optimal = optimal_mapping(weights)

        phi_optimal = compute_phi(
            weights,
            optimal
        )

        gain = (
            (phi_initial - phi_local)
            /
            phi_initial
        ) * 100

        gap = (
            (phi_local - phi_optimal)
            /
            phi_optimal
        ) * 100

        swaps.append(local["swaps"])
        rounds.append(local["rounds"])
        gains.append(gain)
        gaps.append(gap)

    def stats(v):

        mean = np.mean(v)

        std = np.std(v, ddof=1)

        ci95 = 1.96 * std / np.sqrt(runs)

        return mean, std, ci95

    swaps_mean, swaps_std, swaps_ci95 = stats(swaps)
    rounds_mean, rounds_std, rounds_ci95 = stats(rounds)
    gain_mean, gain_std, gain_ci95 = stats(gains)
    gap_mean, gap_std, gap_ci95 = stats(gaps)

    return {

        "swaps_mean": swaps_mean,
        "swaps_std": swaps_std,
        "swaps_ci95": swaps_ci95,

        "rounds_mean": rounds_mean,
        "rounds_std": rounds_std,
        "rounds_ci95": rounds_ci95,

        "gain_mean": gain_mean,
        "gain_std": gain_std,
        "gain_ci95": gain_ci95,

        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "gap_ci95": gap_ci95
    }


# =====================================================
# GRÁFICOS
# =====================================================

def plot_metric(
        df,
        metric,
        ci_metric,
        ylabel,
        filename):

    plt.figure(figsize=(8,5))

    for dist in df["distribution"].unique():

        subset = (
            df[df["distribution"] == dist]
            .sort_values("n")
        )

        x = subset["n"].values

        mean = subset[metric].values

        ci = subset[ci_metric].values

        plt.plot(
            x,
            mean,
            marker="o",
            label=dist
        )

        plt.fill_between(
            x,
            mean - ci,
            mean + ci,
            alpha=0.2
        )

    plt.xscale("log", base=2)

    plt.xlabel("System size (n)")
    plt.ylabel(ylabel)

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# =====================================================
# EXPERIMENTO COMPLETO
# =====================================================

def run_experiment():

    sizes = [
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024
    ]

    runs = 100

    distributions = {
        "Random": random_weights,
        "Zipf": zipf_weights,
        "Concentrated": concentrated_weights
    }

    rows = []

    for dist_name, generator in distributions.items():

        print()
        print("="*80)
        print(dist_name)
        print("="*80)

        for n in sizes:

            r = evaluate_size(
                n,
                generator,
                runs
            )

            print(
                f"n={n:4d} | "
                f"rounds={r['rounds_mean']:.2f} ± {r['rounds_ci95']:.2f} | "
                f"swaps={r['swaps_mean']:.2f} ± {r['swaps_ci95']:.2f} | "
                f"gain={r['gain_mean']:.2f}% ± {r['gain_ci95']:.2f} | "
                f"gap={r['gap_mean']:.2f}% ± {r['gap_ci95']:.2f}"
            )

            rows.append({
                "distribution": dist_name,
                "n": n,

                "rounds_mean":
                    r["rounds_mean"],
                "rounds_std":
                    r["rounds_std"],
                "swaps_mean":
                    r["swaps_mean"],
                "swaps_std":
                    r["swaps_std"],
                "gain_mean":
                    r["gain_mean"],
                "gain_std":
                    r["gain_std"],
                "gap_mean":
                    r["gap_mean"],
                "gap_std":
                    r["gap_std"],
                "rounds_ci95": 
                    r["rounds_ci95"],
                "swaps_ci95": 
                    r["swaps_ci95"],
                "gain_ci95": 
                    r["gain_ci95"],
                "gap_ci95": 
                    r["gap_ci95"],
            })

    df = pd.DataFrame(rows)

    df.to_csv(
        "impact_vcube_d.csv",
        index=False
    )

    plot_metric(
        df,
        "rounds_mean",
        "rounds_ci95",
        "Rounds until convergence",
        "rounds.png"
    )

    plot_metric(
        df,
        "swaps_mean",
        "swaps_ci95",
        "Number of swaps",
        "swaps.png"
    )

    plot_metric(
        df,
        "gain_mean",
        "gain_ci95",
        "Gain in Phi (%)",
        "gain_phi.png"
    )

    plot_metric(
        df,
        "gap_mean",
        "gap_ci95",
        "Gap to global optimum (%)",
        "gap.png"
    )

    return df


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    run_experiment()
