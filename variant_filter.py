"""
Telling real within-host variants from sequencing artefacts: a simulated study.

Why I wrote this
----------------
Consensus sequences average over the viral population inside a host, so
within-host evolution shows up below the consensus, in minor variants. The
practical difficulty is that a mutation seen in 2% of reads can be real or can
be an artefact of PCR, sequencing or amplicon design. A 2026 preprint from the
Lythgoe group and collaborators addresses exactly this (Hamilton et al.,
"Identification and Masking of Artefactual and Misleading Within-Host Variants
in Deep-Sequencing SARS-CoV-2 Data").

I wanted to understand why a frequency threshold alone is a weak filter, so I
built a simulation where I know the ground truth and compared a frequency
cutoff against models trained on per-site features.

What this is and is not
-----------------------
The data are SIMULATED. I wrote the generating process myself from what is
known about common artefact signatures, so any accuracy reported here measures
whether a model can recover structure I put in by hand. It says nothing about
performance on real sequencing data and is not a validation of the approach.

The part I did not expect
-------------------------
My first model treated strand balance and read position as useless: removing
them cost no accuracy at all. They are not useless. Both are symmetric around
0.5, with artefacts at BOTH extremes and real variants in the middle, so their
relationship to the label is U-shaped. A linear model cannot represent that,
and reports a coefficient near zero. Two fixes recover the signal: encode the
distance from the centre, or use a model that is not linear. Section 4 below
compares the three.

Features, and why each one carries signal
-----------------------------------------
  frequency            artefacts concentrate at low frequency
  depth                confidence in the frequency estimate
  strand_balance       a real variant appears on both strands in similar
                       proportion; many artefacts are strongly strand-biased
  mean_base_quality    miscalled bases tend to have lower quality scores
  mean_read_position   errors cluster near read ends; real variants do not
  homopolymer_length   polymerase slippage in homopolymer runs causes errors
  near_amplicon_edge   primer regions and amplicon boundaries are error-prone

Author: Marc-Antoine Dereumaux
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
# Simulated data
# ---------------------------------------------------------------------------

def simulate_sites(n_real=3000, n_artefact=9000):
    """Generate candidate variant calls with known labels.

    Artefacts outnumber real variants 3 to 1 here, which is the regime that
    makes the problem hard: a filter that is right on average can still be
    swamped by false positives.
    """
    # --- real within-host variants -------------------------------------
    real = {
        # A mixture: most real minor variants are rare, a few are rising.
        "frequency": np.concatenate([
            RNG.beta(1.2, 40.0, size=int(n_real * 0.75)),
            RNG.beta(2.0, 8.0, size=n_real - int(n_real * 0.75)),
        ]),
        "depth": RNG.lognormal(mean=6.9, sigma=0.5, size=n_real),
        # Balanced across strands, because a real variant is in the template.
        "strand_balance": np.clip(RNG.normal(0.50, 0.055, size=n_real), 0.0, 1.0),
        "mean_base_quality": np.clip(RNG.normal(36.0, 2.0, size=n_real), 10, 41),
        # No reason to sit at one end of a read.
        "mean_read_position": np.clip(RNG.normal(0.50, 0.10, size=n_real), 0.0, 1.0),
        "homopolymer_length": RNG.choice([1, 2, 3, 4], size=n_real, p=[0.58, 0.27, 0.10, 0.05]),
        "near_amplicon_edge": RNG.random(n_real) < 0.06,
    }

    # --- artefacts -----------------------------------------------------
    half = n_artefact // 2
    artefact = {
        # Overwhelmingly rare, which is why a frequency cutoff is tempting --
        # and why it fails on the rare real variants.
        "frequency": RNG.beta(0.9, 60.0, size=n_artefact),
        "depth": RNG.lognormal(mean=6.6, sigma=0.7, size=n_artefact),
        # Strand bias: a mixture pushed towards 0 and towards 1.
        "strand_balance": np.clip(
            np.concatenate([
                RNG.beta(1.4, 6.0, size=half),
                1.0 - RNG.beta(1.4, 6.0, size=n_artefact - half),
            ]),
            0.0, 1.0,
        ),
        "mean_base_quality": np.clip(RNG.normal(28.0, 4.5, size=n_artefact), 10, 41),
        # Clustered near read ends, again at both ends.
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
    """Append |feature - 0.5| for the two features that are symmetric.

    This is the whole fix for the linear model: it converts "far from the
    middle in either direction" into a single increasing quantity, which a
    linear decision boundary can use.
    """
    extra = [np.abs(X[:, FEATURES.index(name)] - 0.5) for name in SYMMETRIC]
    return np.column_stack([X] + extra)


ENGINEERED = FEATURES + [f"{name}_distance_from_centre" for name in SYMMETRIC]


# ---------------------------------------------------------------------------
# Baseline and models
# ---------------------------------------------------------------------------

def frequency_threshold_baseline(X_test, y_test, thresholds=(0.01, 0.02, 0.03, 0.05, 0.10)):
    """What a plain frequency cutoff achieves.

    This is the honest comparison, because a cutoff is the simplest thing one
    would do and it is still common practice. Precision and recall are
    reported separately, since a strict cutoff looks precise only because it
    discards most real variants.
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
    """Drop one feature at a time and measure the loss.

    A feature whose removal costs nothing is not contributing to THIS model,
    which is not the same as carrying no information.
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
    """Precision when the model is tuned to a given recall.

    Comparing two filters at the same sensitivity is the only fair way; a
    model tuned to a different operating point can be made to look better for
    free.
    """
    positives = np.sort(scores[y_test == 1])[::-1]
    index = min(int(np.ceil(target_recall * positives.size)) - 1, positives.size - 1)
    cutoff = positives[max(index, 0)]
    predicted = scores >= cutoff
    tp = float(((predicted == 1) & (y_test == 1)).sum())
    fp = float(((predicted == 1) & (y_test == 0)).sum())
    return tp / (tp + fp) if (tp + fp) else 0.0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    print("Real within-host variants vs sequencing artefacts (SIMULATED data)")
    print("=" * 72)
    print("The data are simulated by me, so the scores below measure whether a")
    print("model can recover structure I built in. They are not a measurement.")

    X, y = simulate_sites()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )
    print(f"\n{int(y.sum())} real variants, {int((y == 0).sum())} artefacts;"
          f" {y_test.size} held out.")

    # --- 1. the cutoff everyone starts with ----------------------------
    print("\n1. A frequency cutoff on its own")
    print(f"   {'cutoff':>8} {'precision':>10} {'recall':>8} {'F1':>7} {'calls':>7}")
    baseline = frequency_threshold_baseline(X_test, y_test)
    for row in baseline:
        print(f"   {row['threshold']:>8.2f} {row['precision']:>10.3f}"
              f" {row['recall']:>8.3f} {row['f1']:>7.3f} {row['n_called']:>7}")
    best_cutoff = max(baseline, key=lambda r: r["f1"])
    print(f"   Best F1 at {best_cutoff['threshold']:.2f}: it keeps"
          f" {best_cutoff['recall']:.0%} of real variants and is right"
          f" {best_cutoff['precision']:.0%} of the time.")

    # --- 2. linear model on raw features -------------------------------
    linear = fit_logistic(X_train, y_train)
    linear_metrics, linear_scores = evaluate(linear, X_test, y_test)
    print("\n2. Logistic regression, raw features")
    print(f"   ROC AUC {linear_metrics['roc_auc']:.3f}    average precision"
          f" {linear_metrics['average_precision']:.3f}"
          f"   (class rate {y_test.mean():.3f})")

    linear_ablation = ablation(fit_logistic, FEATURES, X_train, y_train,
                               X_test, y_test, linear_metrics["roc_auc"])
    print(f"   {'feature':>22} {'AUC without':>12} {'cost':>7}")
    for row in linear_ablation:
        print(f"   {row['feature']:>22} {row['auc_without']:>12.3f}"
              f" {row['auc_drop']:>7.3f}")
    print("   Strand balance and read position cost nothing to remove, which")
    print("   is suspicious: I built signal into both of them.")

    # --- 3. the diagnosis ----------------------------------------------
    print("\n3. Why: both are U-shaped, and a linear model cannot see that")
    print(f"   {'feature':>20} {'real: mean':>11} {'artefact: mean':>15}"
          f" {'real |d-0.5|':>13} {'artefact |d-0.5|':>17}")
    for name in SYMMETRIC:
        column = X[:, FEATURES.index(name)]
        print(f"   {name:>20} {column[y == 1].mean():>11.3f}"
              f" {column[y == 0].mean():>15.3f}"
              f" {np.abs(column[y == 1] - 0.5).mean():>13.3f}"
              f" {np.abs(column[y == 0] - 0.5).mean():>17.3f}")
    print("   The raw means are nearly identical, because the artefacts sit at")
    print("   both extremes and cancel out. The distance from the centre")
    print("   separates the classes cleanly.")

    # --- 4. two ways to fix it -----------------------------------------
    Xe_train, Xe_test = add_centred_features(X_train), add_centred_features(X_test)
    engineered = fit_logistic(Xe_train, y_train)
    engineered_metrics, engineered_scores = evaluate(engineered, Xe_test, y_test)

    trees = fit_boosted_trees(X_train, y_train)
    tree_metrics, tree_scores = evaluate(trees, X_test, y_test)

    print("\n4. Two fixes")
    print(f"   {'model':>34} {'ROC AUC':>9} {'avg prec':>10}")
    print(f"   {'logistic, raw features':>34} {linear_metrics['roc_auc']:>9.3f}"
          f" {linear_metrics['average_precision']:>10.3f}")
    print(f"   {'logistic + distance from centre':>34}"
          f" {engineered_metrics['roc_auc']:>9.3f}"
          f" {engineered_metrics['average_precision']:>10.3f}")
    print(f"   {'gradient-boosted trees, raw':>34} {tree_metrics['roc_auc']:>9.3f}"
          f" {tree_metrics['average_precision']:>10.3f}")
    print("   Feature engineering and a non-linear model buy roughly the same")
    print("   thing here. The engineered version is the one I would keep: it")
    print("   stays readable, and each coefficient still means something.")

    print("\n   Coefficients of the engineered model (standardised):")
    coefficients = engineered.named_steps["logisticregression"].coef_[0]
    for name, value in sorted(zip(ENGINEERED, coefficients), key=lambda p: -abs(p[1])):
        print(f"   {name:>34} {value:>+8.3f}")

    # --- 5. fair comparison --------------------------------------------
    target_recall = best_cutoff["recall"]
    cutoff_precision = best_cutoff["precision"]
    model_precision = precision_at_recall(engineered_scores, y_test, target_recall)
    print(f"\n5. At equal recall ({target_recall:.3f})")
    print(f"   frequency cutoff {best_cutoff['threshold']:.2f}   precision"
          f" {cutoff_precision:.3f}")
    print(f"   engineered model         precision {model_precision:.3f}")
    print(f"   false-positive rate among calls divided by"
          f" {(1 - cutoff_precision) / max(1 - model_precision, 1e-9):.1f}")

    print("\n6. What I take from this")
    print("   Frequency alone cannot separate the classes, because the real")
    print("   variants that matter most -- the rare ones, early in their rise --")
    print("   live in the same frequency range as the artefacts. The signal is")
    print("   in HOW a variant was observed, not how often.")
    print("   The lesson I did not expect is about my own method: a feature can")
    print("   look worthless because the model cannot express its shape. I would")
    print("   now look at the distributions before trusting a coefficient.")
    print("\n   One number here should not be believed, and it is the best one.")
    print(f"   An AUC of {engineered_metrics['roc_auc']:.3f} means my two classes are far more")
    print("   separable than anything real, because I drew them from distinct")
    print("   distributions with no overlap in the tails and no correlated")
    print("   noise. The ranking of the approaches is the part I would defend;")
    print("   the absolute scores are an artefact of my own generating process.")
    print("\n   What I would want to know next: whether these per-site features")
    print("   are available in the ONS-CIS pipeline output, and how a model")
    print("   trained on one protocol transfers to another, since amplicon and")
    print("   capture data should carry different artefact signatures.")

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
    print("\nWrote variant_filter_results.json")


if __name__ == "__main__":
    main()
