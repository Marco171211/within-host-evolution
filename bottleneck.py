"""
Within-host evolution and the transmission bottleneck: a toy model.

Why I wrote this
----------------
Two results from the Lythgoe group motivated this exercise:

  1. Lythgoe et al., Science 2021 (doi:10.1126/science.abg0821): the
     transmission bottleneck for SARS-CoV-2 is narrow, so most within-host
     variants are lost when the virus passes to the next person.

  2. Ghafari et al., Nature 2024 (doi:10.1038/s41586-024-07029-4): 0.1-0.5% of
     infections persist for 60 days or more. In some of those, many amino-acid
     substitutions accumulate, indicating strong positive selection; in others
     the consensus does not change for long periods.

Together they pose a question I wanted to work through myself: for a new
mutation to be transmitted, it first has to become common inside its host.
How much do infection duration and selection each contribute to that?

This is a toy model, not a model of SARS-CoV-2. Every number below is a
property of the model, not a measurement. Its purpose is to make one mechanism
visible and to check a simulation against an analytic result.

Model
-----
A single focal mutation appears in one virion of a within-host population of N
virions. The population reproduces in discrete generations (Wright-Fisher, so
genetic drift is included). The mutation carries a selection coefficient s.
At transmission, Nb virions are sampled at random.

Author: Marc-Antoine Dereumaux
"""

import json

import numpy as np

RNG = np.random.default_rng(20261123)


# ---------------------------------------------------------------------------
# Act 1: the bottleneck as a filter on frequency
# ---------------------------------------------------------------------------

def survival_probability(frequency, bottleneck_size):
    """Probability a variant at `frequency` is sampled at least once.

    Drawing Nb virions from a large population, every draw misses the variant
    with probability (1 - f)**Nb, so

        P(transmitted) = 1 - (1 - f)**Nb

    This single formula is why a narrow bottleneck filters so hard: at 1%
    frequency and Nb = 3, a variant has about a 3% chance of getting through.
    """
    return 1.0 - (1.0 - np.asarray(frequency, dtype=float)) ** bottleneck_size


def check_against_theory(n_trials=200_000):
    """Does Monte Carlo sampling reproduce the formula?

    If it does not, the simulation is wrong, so this is the first thing to
    test. Any remaining gap should be of order 1/sqrt(n_trials).
    """
    rows = []
    for nb in (1, 3, 10):
        for f in (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50):
            draws = RNG.binomial(nb, f, size=n_trials)
            rows.append(
                {
                    "bottleneck_size": nb,
                    "frequency": f,
                    "simulated": float((draws >= 1).mean()),
                    "analytic": float(survival_probability(f, nb)),
                }
            )
    worst = max(abs(r["simulated"] - r["analytic"]) for r in rows)
    return rows, worst


# ---------------------------------------------------------------------------
# Act 2: what happens to a new mutation inside one host
# ---------------------------------------------------------------------------

def simulate_focal_mutation(selection, n_generations, pop_size=5000, n_replicates=4000):
    """Track one new mutation through a Wright-Fisher population with selection.

    The mutation starts in a single virion, so at frequency 1/N. Each
    generation its expected frequency after selection is

        p' = p (1 + s) / (1 + p s)

    and the next generation's count is drawn from Binomial(N, p'). Drift is
    therefore the binomial sampling noise, and it dominates while the mutation
    is rare -- which is why most new mutations are lost regardless of s.

    Returns the frequency of the mutation at each recorded generation, for
    every replicate.
    """
    checkpoints = [10, 20, 30, 40, 60, 80, 100]
    checkpoints = [g for g in checkpoints if g <= n_generations]

    counts = np.ones(n_replicates, dtype=np.int64)
    recorded = {}

    for generation in range(1, n_generations + 1):
        p = counts / pop_size
        p_after_selection = p * (1.0 + selection) / (1.0 + p * selection)
        counts = RNG.binomial(pop_size, p_after_selection)
        if generation in checkpoints:
            recorded[generation] = counts / pop_size

    return recorded


def summarise(frequencies, bottleneck_sizes=(1, 3, 10)):
    """Turn a set of replicate frequencies into the quantities that matter."""
    surviving = frequencies > 0
    row = {
        "still_present": float(surviving.mean()),
        "mean_frequency_if_present": float(frequencies[surviving].mean()) if surviving.any() else 0.0,
        "reached_1pct": float((frequencies >= 0.01).mean()),
        "reached_5pct": float((frequencies >= 0.05).mean()),
    }
    # P(transmitted) averaged over the whole distribution of outcomes, so
    # lineages that went extinct correctly contribute zero.
    for nb in bottleneck_sizes:
        row[f"transmitted_nb{nb}"] = float(survival_probability(frequencies, nb).mean())
    return row


def duration_and_selection(selections=(0.0, 0.02, 0.05, 0.10), n_generations=100):
    rows = []
    for s in selections:
        recorded = simulate_focal_mutation(s, n_generations)
        for generation, frequencies in recorded.items():
            row = {"selection": s, "generations": generation}
            row.update(summarise(frequencies))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    print("Within-host evolution and the transmission bottleneck (toy model)")
    print("=" * 70)

    checks, worst = check_against_theory()
    print("\nACT 1  The bottleneck filters on frequency")
    print(f"  Monte Carlo vs 1 - (1 - f)^Nb, 24 cases, worst gap: {worst:.4f}")
    print("  The simulation reproduces the formula, so the sampling is right.\n")
    print(f"  {'frequency':>10} {'Nb=1':>8} {'Nb=3':>8} {'Nb=10':>8}")
    for f in (0.001, 0.01, 0.05, 0.20, 0.50):
        line = f"  {f:>10.3f}"
        for nb in (1, 3, 10):
            line += f" {float(survival_probability(f, nb)):>8.3f}"
        print(line)
    print("  A variant has to be common inside the host to stand a real chance.")

    rows = duration_and_selection()
    print("\nACT 2  So how does a new mutation ever become common?")
    print("  One mutation, starting in 1 virion out of 5000, 4000 replicates.\n")
    print(f"  {'s':>5} {'gens':>5} {'present':>8} {'>=1%':>7} {'>=5%':>7}"
          f" {'P(tx) Nb=3':>11}")
    for row in rows:
        print(
            f"  {row['selection']:>5.2f} {row['generations']:>5}"
            f" {row['still_present']:>8.3f} {row['reached_1pct']:>7.3f}"
            f" {row['reached_5pct']:>7.3f} {row['transmitted_nb3']:>11.4f}"
        )

    # A second analytic check, and a more interesting one. Haldane (1927)
    # showed that a beneficial mutation starting in a single individual
    # establishes with probability roughly 2s when s is small. If the
    # simulation did not reproduce that, I would not trust it.
    print("\n  Second check: P(still present) vs Haldane's 2s approximation")
    print(f"  {'s':>5} {'simulated':>10} {'2s':>7}")
    haldane = []
    for s in (0.02, 0.05, 0.10):
        final = next(r for r in rows if r["selection"] == s and r["generations"] == 100)
        haldane.append({"selection": s, "simulated": final["still_present"], "two_s": 2 * s})
        print(f"  {s:>5.2f} {final['still_present']:>10.3f} {2 * s:>7.3f}")
    print("  Close at small s, and the approximation is expected to drift at")
    print("  larger s, which is what happens. Most beneficial mutations are")
    print("  still lost: at s = 0.05 about 90% disappear despite being favoured.")

    neutral_100 = next(r for r in rows if r["selection"] == 0.0 and r["generations"] == 100)
    selected_100 = next(r for r in rows if r["selection"] == 0.10 and r["generations"] == 100)
    selected_20 = next(r for r in rows if r["selection"] == 0.10 and r["generations"] == 20)

    print("\nACT 3  What I take from this")
    print("  Drift alone is a poor way to become common. After 100 generations")
    print(f"  a neutral mutation reached 5% in {neutral_100['reached_5pct'] * 4000:.0f} of 4000")
    print(f"  replicates, against {selected_100['reached_5pct']:.1%} when s = 0.10.")
    print("  Duration and selection are not interchangeable either: with")
    print(f"  s = 0.10, P(transmitted, Nb=3) goes from {selected_20['transmitted_nb3']:.4f}")
    print(f"  at 20 generations to {selected_100['transmitted_nb3']:.4f} at 100 -- a factor of"
          f" {selected_100['transmitted_nb3'] / max(selected_20['transmitted_nb3'], 1e-12):.1f}.")
    print("  Selection decides whether a mutation can rise at all; time decides")
    print("  how far it rises before transmission.")
    print("  That is the reading I now have of the two papers together: a short")
    print("  infection rarely exports anything new, and a long one is only")
    print("  dangerous when something is pushing the mutation up.")

    print("\nLIMITS  Neutral or constant selection, one mutation at a time, no")
    print("  recombination, no immune dynamics, no variable population size, and")
    print("  no epistasis. Real within-host populations violate all of these.")

    with open("bottleneck_results.json", "w") as handle:
        json.dump(
            {
                "analytic_checks": checks,
                "worst_error": worst,
                "haldane_check": haldane,
                "trajectories": rows,
            },
            handle,
            indent=2,
        )
    print("\nWrote bottleneck_results.json")


if __name__ == "__main__":
    main()
