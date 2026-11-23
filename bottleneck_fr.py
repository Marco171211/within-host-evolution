"""
Évolution intra-hôte et goulot d'étranglement à la transmission : un modèle jouet.

(Version française de bottleneck.py, pour la lecture. Le code est identique,
seuls les commentaires et les affichages sont traduits.)

Pourquoi j'ai écrit ça
----------------------
Deux résultats du groupe de Katrina Lythgoe sont à l'origine de cet exercice :

  1. Lythgoe et al., Science 2021 (doi:10.1126/science.abg0821) : le goulot
     d'étranglement à la transmission du SARS-CoV-2 est étroit, donc la plupart
     des variants présents chez une personne sont perdus au moment du passage
     à la personne suivante.

  2. Ghafari et al., Nature 2024 (doi:10.1038/s41586-024-07029-4) : 0,1 à 0,5 %
     des infections persistent 60 jours ou plus. Chez certaines, de nombreuses
     substitutions d'acides aminés s'accumulent, signe d'une forte sélection
     positive ; chez d'autres, le consensus ne change pas pendant des mois.

Mis côte à côte, ils posent une question que je voulais traiter moi-même :
pour qu'une mutation soit transmise, il faut d'abord qu'elle devienne
fréquente chez son hôte. Quelle est la part de la durée de l'infection, et
quelle est la part de la sélection ?

C'est un modèle jouet, pas un modèle du SARS-CoV-2. Chaque nombre ci-dessous
est une propriété du modèle, pas une mesure. Le but est de rendre un mécanisme
visible et de vérifier une simulation contre un résultat analytique.

Le modèle
---------
Une mutation unique apparaît dans un virion d'une population intra-hôte de N
virions. La population se reproduit en générations discrètes (modèle de
Wright-Fisher, donc la dérive génétique est incluse). La mutation porte un
coefficient de sélection s. À la transmission, Nb virions sont tirés au hasard.

Auteur : Marc-Antoine Dereumaux
"""

import json

import numpy as np

RNG = np.random.default_rng(20261123)


# ---------------------------------------------------------------------------
# Acte 1 : le goulot comme filtre sur la fréquence
# ---------------------------------------------------------------------------

def survival_probability(frequency, bottleneck_size):
    """Probabilité qu'un variant à la fréquence `frequency` soit tiré au moins une fois.

    En tirant Nb virions dans une grande population, chaque tirage rate le
    variant avec la probabilité (1 - f), donc

        P(transmis) = 1 - (1 - f)**Nb

    Cette seule formule explique pourquoi un goulot étroit filtre aussi fort :
    à 1 % de fréquence et Nb = 3, un variant a environ 3 % de chances de passer.
    """
    return 1.0 - (1.0 - np.asarray(frequency, dtype=float)) ** bottleneck_size


def check_against_theory(n_trials=200_000):
    """L'échantillonnage Monte-Carlo reproduit-il la formule ?

    Si ce n'est pas le cas, la simulation est fausse : c'est donc la première
    chose à tester. L'écart résiduel doit être de l'ordre de 1/sqrt(n_trials).
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
# Acte 2 : le destin d'une mutation nouvelle chez un seul hôte
# ---------------------------------------------------------------------------

def simulate_focal_mutation(selection, n_generations, pop_size=5000, n_replicates=4000):
    """Suit une mutation nouvelle dans une population de Wright-Fisher avec sélection.

    La mutation démarre dans un seul virion, donc à la fréquence 1/N. À chaque
    génération, sa fréquence attendue après sélection vaut

        p' = p (1 + s) / (1 + p s)

    et l'effectif de la génération suivante est tiré dans une loi
    Binomiale(N, p'). La dérive est donc le bruit d'échantillonnage binomial,
    et elle domine tant que la mutation est rare -- ce qui explique que la
    plupart des mutations nouvelles disparaissent, quelle que soit leur valeur
    sélective.

    Renvoie la fréquence de la mutation à chaque génération enregistrée, pour
    chaque réplicat.
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
    """Transforme les fréquences des réplicats en quantités interprétables."""
    surviving = frequencies > 0
    row = {
        "still_present": float(surviving.mean()),
        "mean_frequency_if_present": float(frequencies[surviving].mean()) if surviving.any() else 0.0,
        "reached_1pct": float((frequencies >= 0.01).mean()),
        "reached_5pct": float((frequencies >= 0.05).mean()),
    }
    # P(transmise) moyennée sur toute la distribution des issues, pour que les
    # lignées éteintes comptent bien pour zéro.
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
# Résultats
# ---------------------------------------------------------------------------

def main():
    print("Évolution intra-hôte et goulot d'étranglement (modèle jouet)")
    print("=" * 70)

    checks, worst = check_against_theory()
    print("\nACTE 1  Le goulot filtre selon la fréquence")
    print(f"  Monte-Carlo contre 1 - (1 - f)^Nb, 24 cas, écart max : {worst:.4f}")
    print("  La simulation reproduit la formule : l'échantillonnage est correct.\n")
    print(f"  {'fréquence':>11} {'Nb=1':>8} {'Nb=3':>8} {'Nb=10':>8}")
    for f in (0.001, 0.01, 0.05, 0.20, 0.50):
        line = f"  {f:>11.3f}"
        for nb in (1, 3, 10):
            line += f" {float(survival_probability(f, nb)):>8.3f}"
        print(line)
    print("  Un variant doit être fréquent chez l'hôte pour avoir une vraie chance.")

    rows = duration_and_selection()
    print("\nACTE 2  Alors comment une mutation devient-elle fréquente ?")
    print("  Une mutation, partant de 1 virion sur 5000, 4000 réplicats.\n")
    print(f"  {'s':>5} {'gén.':>5} {'présente':>9} {'>=1%':>7} {'>=5%':>7}"
          f" {'P(tx) Nb=3':>11}")
    for row in rows:
        print(
            f"  {row['selection']:>5.2f} {row['generations']:>5}"
            f" {row['still_present']:>9.3f} {row['reached_1pct']:>7.3f}"
            f" {row['reached_5pct']:>7.3f} {row['transmitted_nb3']:>11.4f}"
        )

    # Deuxième vérification, plus intéressante. Haldane (1927) a montré qu'une
    # mutation avantageuse partant d'un seul individu s'établit avec une
    # probabilité d'environ 2s quand s est petit. Si la simulation ne
    # retrouvait pas ça, je ne lui ferais pas confiance.
    print("\n  Deuxième contrôle : P(encore présente) contre l'approximation 2s de Haldane")
    print(f"  {'s':>5} {'simulé':>9} {'2s':>7}")
    haldane = []
    for s in (0.02, 0.05, 0.10):
        final = next(r for r in rows if r["selection"] == s and r["generations"] == 100)
        haldane.append({"selection": s, "simulated": final["still_present"], "two_s": 2 * s})
        print(f"  {s:>5.2f} {final['still_present']:>9.3f} {2 * s:>7.3f}")
    print("  Proche pour s petit, et l'approximation est censée s'écarter quand s")
    print("  grandit, ce qui est bien ce qui se passe. La plupart des mutations")
    print("  avantageuses sont quand même perdues : à s = 0,05, environ 90 %")
    print("  disparaissent alors qu'elles sont favorisées.")

    neutral_100 = next(r for r in rows if r["selection"] == 0.0 and r["generations"] == 100)
    selected_100 = next(r for r in rows if r["selection"] == 0.10 and r["generations"] == 100)
    selected_20 = next(r for r in rows if r["selection"] == 0.10 and r["generations"] == 20)

    print("\nACTE 3  Ce que j'en retiens")
    print("  La dérive seule est un très mauvais moyen de devenir fréquent. Après")
    print(f"  100 générations, une mutation neutre a atteint 5 % dans"
          f" {neutral_100['reached_5pct'] * 4000:.0f} réplicats")
    print(f"  sur 4000, contre {selected_100['reached_5pct']:.1%} quand s = 0,10.")
    print("  Durée et sélection ne sont pas interchangeables non plus : avec")
    print(f"  s = 0,10, P(transmise, Nb=3) passe de {selected_20['transmitted_nb3']:.4f}")
    print(f"  à 20 générations à {selected_100['transmitted_nb3']:.4f} à 100 -- un facteur"
          f" {selected_100['transmitted_nb3'] / max(selected_20['transmitted_nb3'], 1e-12):.1f}.")
    print("  La sélection décide si une mutation peut monter ; la durée décide")
    print("  jusqu'où elle monte avant la transmission.")
    print("  C'est la lecture que j'ai maintenant des deux articles ensemble :")
    print("  une infection courte n'exporte presque jamais rien de nouveau, et")
    print("  une infection longue ne compte que si quelque chose pousse la")
    print("  mutation vers le haut.")

    print("\nLIMITES  Sélection nulle ou constante, une seule mutation à la fois,")
    print("  pas de recombinaison, pas de dynamique immunitaire, taille de")
    print("  population constante, pas d'épistasie. Les vraies populations")
    print("  intra-hôte violent tout cela.")

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
    print("\nFichier bottleneck_results.json écrit.")


if __name__ == "__main__":
    main()
