"""
Distinguer les vrais variants intra-hôte des artéfacts de séquençage.

(Version française de variant_filter.py, pour la lecture. Le code est
identique, seuls les commentaires et les affichages sont traduits.)

Pourquoi j'ai écrit ça
----------------------
Une séquence consensus moyenne la population virale présente chez un hôte :
l'évolution intra-hôte se joue donc en dessous du consensus, dans les variants
minoritaires. La difficulté pratique est qu'une mutation vue dans 2 % des
lectures peut être réelle ou être un artéfact de PCR, de séquençage ou de
conception des amplicons. Un preprint de 2026 du groupe de Katrina Lythgoe et
de collaborateurs porte exactement là-dessus (Hamilton et al., « Identification
and Masking of Artefactual and Misleading Within-Host Variants in
Deep-Sequencing SARS-CoV-2 Data »).

Je voulais comprendre pourquoi un simple seuil de fréquence est un filtre
faible. J'ai donc construit une simulation dont je connais la vérité terrain,
et comparé un seuil à des modèles entraînés sur des variables par site.

Ce que c'est, et ce que ce n'est pas
------------------------------------
Les données sont SIMULÉES. J'ai écrit moi-même le processus qui les engendre, à
partir de ce que l'on sait des signatures d'artéfacts courantes. Toute
performance rapportée ici mesure donc si un modèle retrouve une structure que
j'ai mise à la main. Cela ne dit rien des performances sur de vraies données de
séquençage, et ce n'est pas une validation de l'approche.

La partie que je n'avais pas prévue
-----------------------------------
Mon premier modèle traitait le biais de brin et la position dans la lecture
comme inutiles : les retirer ne coûtait aucune performance. Ils ne sont pas
inutiles. Les deux sont symétriques autour de 0,5, avec les artéfacts aux DEUX
extrêmes et les vrais variants au milieu : leur relation à l'étiquette est donc
en U. Un modèle linéaire ne sait pas représenter ça et renvoie un coefficient
proche de zéro. Deux corrections récupèrent le signal : encoder la distance au
centre, ou utiliser un modèle non linéaire. La section 4 compare les trois.

Les variables, et le signal que chacune porte
---------------------------------------------
  frequency            les artéfacts se concentrent aux basses fréquences
  depth                la confiance dans l'estimation de fréquence
  strand_balance       un vrai variant apparaît sur les deux brins dans des
                       proportions voisines ; beaucoup d'artéfacts sont
                       fortement biaisés vers un brin
  mean_base_quality    les bases mal lues ont des scores de qualité plus bas
  mean_read_position   les erreurs se groupent aux extrémités des lectures,
                       pas les vrais variants
  homopolymer_length   le glissement de la polymérase dans les homopolymères
                       engendre des erreurs
  near_amplicon_edge   les régions d'amorces et les bords d'amplicons sont
                       sujets aux erreurs

Auteur : Marc-Antoine Dereumaux
"""

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = np.random.default_rng(20261123)

FEATURES = [
    "frequency",
    "depth",
    "strand_balance",
    "mean_base_quality",
    "mean_read_position",
    "homopolymer_length",
    "near_amplicon_edge",
]

SYMMETRIC = ["strand_balance", "mean_read_position"]


# ---------------------------------------------------------------------------
# Données simulées
# ---------------------------------------------------------------------------

def simulate_sites(n_real=3000, n_artefact=9000):
    """Engendre des appels de variants candidats dont on connaît l'étiquette.

    Les artéfacts sont 3 fois plus nombreux que les vrais variants, ce qui est
    le régime qui rend le problème difficile : un filtre correct en moyenne
    peut quand même être noyé sous les faux positifs.
    """
    # --- vrais variants intra-hôte -------------------------------------
    real = {
        # Un mélange : la plupart des vrais variants minoritaires sont rares,
        # quelques-uns sont en train de monter.
        "frequency": np.concatenate([
            RNG.beta(1.2, 40.0, size=int(n_real * 0.75)),
            RNG.beta(2.0, 8.0, size=n_real - int(n_real * 0.75)),
        ]),
        "depth": RNG.lognormal(mean=6.9, sigma=0.5, size=n_real),
        # Équilibré entre les brins, parce qu'un vrai variant est dans la matrice.
        "strand_balance": np.clip(RNG.normal(0.50, 0.055, size=n_real), 0.0, 1.0),
        "mean_base_quality": np.clip(RNG.normal(36.0, 2.0, size=n_real), 10, 41),
        # Aucune raison de se trouver à une extrémité des lectures.
        "mean_read_position": np.clip(RNG.normal(0.50, 0.10, size=n_real), 0.0, 1.0),
        "homopolymer_length": RNG.choice([1, 2, 3, 4], size=n_real, p=[0.58, 0.27, 0.10, 0.05]),
        "near_amplicon_edge": RNG.random(n_real) < 0.06,
    }

    # --- artéfacts -----------------------------------------------------
    half = n_artefact // 2
    artefact = {
        # Très majoritairement rares, ce qui rend le seuil de fréquence
        # tentant -- et ce qui le fait échouer sur les vrais variants rares.
        "frequency": RNG.beta(0.9, 60.0, size=n_artefact),
        "depth": RNG.lognormal(mean=6.6, sigma=0.7, size=n_artefact),
        # Biais de brin : un mélange poussé vers 0 et vers 1.
        "strand_balance": np.clip(
            np.concatenate([
                RNG.beta(1.4, 6.0, size=half),
                1.0 - RNG.beta(1.4, 6.0, size=n_artefact - half),
            ]),
            0.0, 1.0,
        ),
        "mean_base_quality": np.clip(RNG.normal(28.0, 4.5, size=n_artefact), 10, 41),
        # Groupés près des extrémités, là aussi des deux côtés.
        "mean_read_position": np.clip(
            np.concatenate([
                RNG.beta(1.6, 6.0, size=half),
                1.0 - RNG.beta(1.6, 6.0, size=n_artefact - half),
            ]),
            0.0, 1.0,
        ),
        "homopolymer_length": RNG.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_artefact,
                                         p=[0.18, 0.17, 0.17, 0.15, 0.12, 0.10, 0.07, 0.04]),
        "near_amplicon_edge": RNG.random(n_artefact) < 0.34,
    }

    X = np.column_stack([
        np.concatenate([real[name], artefact[name]]).astype(float)
        for name in FEATURES
    ])
    y = np.concatenate([np.ones(n_real), np.zeros(n_artefact)])

    order = RNG.permutation(y.size)
    return X[order], y[order]


def add_centred_features(X):
    """Ajoute |variable - 0,5| pour les deux variables symétriques.

    C'est toute la correction du modèle linéaire : on convertit « loin du
    milieu dans un sens ou dans l'autre » en une seule quantité croissante,
    qu'une frontière de décision linéaire sait utiliser.
    """
    extra = [np.abs(X[:, FEATURES.index(name)] - 0.5) for name in SYMMETRIC]
    return np.column_stack([X] + extra)


ENGINEERED = FEATURES + [f"{name}_distance_from_centre" for name in SYMMETRIC]


# ---------------------------------------------------------------------------
# Référence et modèles
# ---------------------------------------------------------------------------

def frequency_threshold_baseline(X_test, y_test, thresholds=(0.01, 0.02, 0.03, 0.05, 0.10)):
    """Ce qu'obtient un simple seuil de fréquence.

    C'est la comparaison honnête, parce qu'un seuil est la chose la plus
    simple que l'on ferait, et que c'est encore une pratique courante. La
    précision et le rappel sont donnés séparément, car un seuil strict ne
    paraît précis que parce qu'il jette la plupart des vrais variants.
    """
    frequency = X_test[:, FEATURES.index("frequency")]
    rows = []
    for threshold in thresholds:
        predicted = frequency >= threshold
        tp = float(((predicted == 1) & (y_test == 1)).sum())
        fp = float(((predicted == 1) & (y_test == 0)).sum())
        n_real = float((y_test == 1).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_real if n_real else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_called": int(predicted.sum()),
        })
    return rows


def fit_logistic(X_train, y_train):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    model.fit(X_train, y_train)
    return model


def fit_boosted_trees(X_train, y_train):
    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.1, max_iter=250, random_state=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
    }, scores


def ablation(fit, feature_names, X_train, y_train, X_test, y_test, full_auc):
    """Retire une variable à la fois et mesure la perte.

    Une variable dont le retrait ne coûte rien ne contribue pas À CE
    MODÈLE-LÀ, ce qui n'est pas la même chose que ne porter aucune information.
    """
    rows = []
    for index, name in enumerate(feature_names):
        keep = [i for i in range(len(feature_names)) if i != index]
        model = fit(X_train[:, keep], y_train)
        auc = roc_auc_score(y_test, model.predict_proba(X_test[:, keep])[:, 1])
        rows.append({"feature": name, "auc_without": float(auc),
                     "auc_drop": float(full_auc - auc)})
    return sorted(rows, key=lambda r: -r["auc_drop"])


def precision_at_recall(scores, y_test, target_recall):
    """Précision du modèle réglé sur un rappel donné.

    Comparer deux filtres à la même sensibilité est la seule façon équitable :
    un modèle réglé sur un autre point de fonctionnement peut être rendu
    meilleur gratuitement.
    """
    positives = np.sort(scores[y_test == 1])[::-1]
    index = min(int(np.ceil(target_recall * positives.size)) - 1, positives.size - 1)
    cutoff = positives[max(index, 0)]
    predicted = scores >= cutoff
    tp = float(((predicted == 1) & (y_test == 1)).sum())
    fp = float(((predicted == 1) & (y_test == 0)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


# ---------------------------------------------------------------------------
# Résultats
# ---------------------------------------------------------------------------

def main():
    print("Vrais variants intra-hôte contre artéfacts (données SIMULÉES)")
    print("=" * 72)
    print("Les données sont simulées par moi : les scores ci-dessous mesurent")
    print("si un modèle retrouve une structure que j'ai construite. Ce ne sont")
    print("pas des mesures.")

    X, y = simulate_sites()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    print(f"\n{int(y.sum())} vrais variants, {int((y == 0).sum())} artéfacts ;"
          f" {y_test.size} mis de côté pour le test.")

    # --- 1. le seuil par lequel tout le monde commence -----------------
    print("\n1. Un seuil de fréquence tout seul")
    print(f"   {'seuil':>8} {'précision':>11} {'rappel':>8} {'F1':>7} {'appels':>8}")
    baseline = frequency_threshold_baseline(X_test, y_test)
    for row in baseline:
        print(f"   {row['threshold']:>8.2f} {row['precision']:>11.3f}"
              f" {row['recall']:>8.3f} {row['f1']:>7.3f} {row['n_called']:>8}")
    best_cutoff = max(baseline, key=lambda r: r["f1"])
    print(f"   Meilleur F1 à {best_cutoff['threshold']:.2f} : il garde"
          f" {best_cutoff['recall']:.0%} des vrais variants et a raison"
          f" {best_cutoff['precision']:.0%} du temps.")

    # --- 2. modèle linéaire sur les variables brutes -------------------
    linear = fit_logistic(X_train, y_train)
    linear_metrics, linear_scores = evaluate(linear, X_test, y_test)
    print("\n2. Régression logistique, variables brutes")
    print(f"   AUC ROC {linear_metrics['roc_auc']:.3f}    précision moyenne"
          f" {linear_metrics['average_precision']:.3f}"
          f"   (taux de classe {y_test.mean():.3f})")

    linear_ablation = ablation(fit_logistic, FEATURES, X_train, y_train,
                               X_test, y_test, linear_metrics["roc_auc"])
    print(f"   {'variable':>22} {'AUC sans':>10} {'coût':>7}")
    for row in linear_ablation:
        print(f"   {row['feature']:>22} {row['auc_without']:>10.3f}"
              f" {row['auc_drop']:>7.3f}")
    print("   Retirer le biais de brin et la position dans la lecture ne coûte")
    print("   rien, ce qui est suspect : j'ai mis du signal dans les deux.")

    # --- 3. le diagnostic ----------------------------------------------
    print("\n3. Pourquoi : les deux sont en U, et un modèle linéaire ne le voit pas")
    print(f"   {'variable':>20} {'vrais : moy.':>13} {'artéf. : moy.':>14}"
          f" {'vrais |d-0,5|':>15} {'artéf. |d-0,5|':>16}")
    for name in SYMMETRIC:
        column = X[:, FEATURES.index(name)]
        print(f"   {name:>20} {column[y == 1].mean():>13.3f}"
              f" {column[y == 0].mean():>14.3f}"
              f" {np.abs(column[y == 1] - 0.5).mean():>15.3f}"
              f" {np.abs(column[y == 0] - 0.5).mean():>16.3f}")
    print("   Les moyennes brutes sont presque identiques, parce que les")
    print("   artéfacts se trouvent aux deux extrêmes et se compensent. La")
    print("   distance au centre sépare nettement les deux classes.")

    # --- 4. deux façons de corriger ------------------------------------
    Xe_train, Xe_test = add_centred_features(X_train), add_centred_features(X_test)
    engineered = fit_logistic(Xe_train, y_train)
    engineered_metrics, engineered_scores = evaluate(engineered, Xe_test, y_test)

    trees = fit_boosted_trees(X_train, y_train)
    tree_metrics, tree_scores = evaluate(trees, X_test, y_test)

    print("\n4. Deux corrections")
    print(f"   {'modèle':>38} {'AUC ROC':>9} {'préc. moy.':>11}")
    print(f"   {'logistique, variables brutes':>38} {linear_metrics['roc_auc']:>9.3f}"
          f" {linear_metrics['average_precision']:>11.3f}")
    print(f"   {'logistique + distance au centre':>38}"
          f" {engineered_metrics['roc_auc']:>9.3f}"
          f" {engineered_metrics['average_precision']:>11.3f}")
    print(f"   {'arbres de gradient, brutes':>38} {tree_metrics['roc_auc']:>9.3f}"
          f" {tree_metrics['average_precision']:>11.3f}")
    print("   La construction de variables et le modèle non linéaire apportent")
    print("   à peu près la même chose ici. Je garderais la version avec")
    print("   variables construites : elle reste lisible, et chaque coefficient")
    print("   veut encore dire quelque chose.")

    print("\n   Coefficients du modèle avec variables construites (standardisés) :")
    coefficients = engineered.named_steps["logisticregression"].coef_[0]
    for name, value in sorted(zip(ENGINEERED, coefficients), key=lambda p: -abs(p[1])):
        print(f"   {name:>40} {value:>+8.3f}")

    # --- 5. comparaison équitable --------------------------------------
    target_recall = best_cutoff["recall"]
    cutoff_precision = best_cutoff["precision"]
    model_precision = precision_at_recall(engineered_scores, y_test, target_recall)
    print(f"\n5. À rappel égal ({target_recall:.3f})")
    print(f"   seuil de fréquence {best_cutoff['threshold']:.2f}   précision"
          f" {cutoff_precision:.3f}")
    print(f"   modèle avec variables construites   précision {model_precision:.3f}")
    print(f"   taux de faux positifs parmi les appels divisé par"
          f" {(1 - cutoff_precision) / max(1 - model_precision, 1e-9):.1f}")

    print("\n6. Ce que j'en retiens")
    print("   La fréquence seule ne sépare pas les classes, parce que les vrais")
    print("   variants qui comptent le plus -- les rares, au début de leur")
    print("   montée -- vivent dans la même plage de fréquence que les")
    print("   artéfacts. Le signal est dans la FAÇON dont un variant a été")
    print("   observé, pas dans sa fréquence.")
    print("   La leçon que je n'attendais pas porte sur ma propre méthode : une")
    print("   variable peut sembler inutile parce que le modèle ne sait pas")
    print("   exprimer sa forme. Je regarderai désormais les distributions avant")
    print("   de faire confiance à un coefficient.")
    print("\n   Un chiffre ici ne doit pas être cru, et c'est le meilleur.")
    print(f"   Une AUC de {engineered_metrics['roc_auc']:.3f} signifie que mes deux classes sont")
    print("   bien plus séparables que la réalité : je les ai tirées de lois")
    print("   distinctes, sans recouvrement dans les queues ni bruit corrélé.")
    print("   Le classement des approches est ce que je défendrais ; les scores")
    print("   absolus sont un artéfact de mon propre générateur de données.")
    print("\n   Ce que je voudrais savoir ensuite : si ces variables par site")
    print("   existent dans la sortie du pipeline de l'enquête ONS-CIS, et")
    print("   comment un modèle entraîné sur un protocole se transfère à un")
    print("   autre, puisque les données d'amplicons et de capture devraient")
    print("   porter des signatures d'artéfacts différentes.")

    fpr_linear, tpr_linear, _ = roc_curve(y_test, linear_scores)
    fpr_eng, tpr_eng, _ = roc_curve(y_test, engineered_scores)
    step_linear = max(len(fpr_linear) // 120, 1)
    step_eng = max(len(fpr_eng) // 120, 1)

    with open("variant_filter_results.json", "w") as handle:
        json.dump(
            {
                "n_real": int(y.sum()),
                "n_artefact": int((y == 0).sum()),
                "class_rate": float(y_test.mean()),
                "models": {
                    "logistic_raw": linear_metrics,
                    "logistic_engineered": engineered_metrics,
                    "boosted_trees": tree_metrics,
                },
                "roc_curves": {
                    "logistic_raw": [
                        {"fpr": float(a), "tpr": float(b)}
                        for a, b in zip(fpr_linear[::step_linear], tpr_linear[::step_linear])
                    ],
                    "logistic_engineered": [
                        {"fpr": float(a), "tpr": float(b)}
                        for a, b in zip(fpr_eng[::step_eng], tpr_eng[::step_eng])
                    ],
                },
                "coefficients": [
                    {"feature": name, "coefficient": float(value)}
                    for name, value in zip(ENGINEERED, coefficients)
                ],
                "linear_ablation": linear_ablation,
                "frequency_baseline": baseline,
                "equal_recall_comparison": {
                    "recall": target_recall,
                    "cutoff": best_cutoff["threshold"],
                    "cutoff_precision": cutoff_precision,
                    "model_precision": model_precision,
                },
                "symmetric_feature_means": [
                    {
                        "feature": name,
                        "real_mean": float(X[y == 1, FEATURES.index(name)].mean()),
                        "artefact_mean": float(X[y == 0, FEATURES.index(name)].mean()),
                        "real_distance": float(np.abs(X[y == 1, FEATURES.index(name)] - 0.5).mean()),
                        "artefact_distance": float(np.abs(X[y == 0, FEATURES.index(name)] - 0.5).mean()),
                    }
                    for name in SYMMETRIC
                ],
            },
            handle,
            indent=2,
        )
    print("\nFichier variant_filter_results.json écrit.")


if __name__ == "__main__":
    main()
