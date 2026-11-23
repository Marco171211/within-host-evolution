# Within-host viral evolution: two small studies

Two short Python studies I wrote to understand two results from the Lythgoe
group at Oxford. Both are toy models on simulated data. Neither is a
contribution to the field; they are how I taught myself what the papers are
actually claiming.

I am a first-year engineering student at Institut Agro Rennes-Angers, on the
statistics track, taking the MITx MicroMasters in Statistics and Data Science
alongside my degree. I wrote this in November 2026.

## 1. `bottleneck.py` — why a new mutation rarely gets transmitted

Motivated by:

- Lythgoe et al., *Science* 2021, [doi:10.1126/science.abg0821](https://doi.org/10.1126/science.abg0821)
  — the SARS-CoV-2 transmission bottleneck is narrow, so most within-host
  variants are lost at transmission.
- Ghafari et al., *Nature* 2024, [doi:10.1038/s41586-024-07029-4](https://doi.org/10.1038/s41586-024-07029-4)
  — 0.1–0.5% of infections persist 60 days or more, with strong positive
  selection in some of them.

The question I wanted to answer for myself: for a new mutation to be
transmitted it must first become common inside its host, so how much do
infection duration and selection each contribute?

A single new mutation is followed through a Wright–Fisher population of 5,000
virions with selection coefficient `s`, over 4,000 replicates, and then through
a transmission bottleneck of `Nb` virions.

**Two checks before trusting any of it.** The Monte Carlo sampling reproduces
`P(transmitted) = 1 − (1 − f)^Nb` to within 0.003 over 24 cases. And the
probability that a beneficial mutation is still present after 100 generations
matches Haldane's classic `2s` approximation (simulated 0.043 / 0.100 / 0.175
against 0.04 / 0.10 / 0.20 for s = 0.02 / 0.05 / 0.10), drifting at larger `s`
as expected.

**What comes out.** A neutral mutation reached 5% frequency in 0 of 4,000
replicates after 100 generations; with `s = 0.10` it did so in 17.4%. And
duration matters on top of that: at `s = 0.10`, the probability of being
transmitted through a bottleneck of 3 rises from 0.004 at 20 generations to
0.170 at 100, a factor of 41. Selection decides whether a mutation can rise at
all; time decides how far it rises before transmission.

**Limits.** Neutral or constant selection, one mutation at a time, no
recombination, no immune dynamics, constant population size, no epistasis.
Real within-host populations violate all of these. Adding immune-driven
time-varying selection is the obvious next step.

## 2. `variant_filter.py` — real minor variants vs sequencing artefacts

Motivated by Hamilton et al. (2026 preprint), *Identification and Masking of
Artefactual and Misleading Within-Host Variants in Deep-Sequencing SARS-CoV-2
Data*, [doi:10.64898/2026.03.19.712895](https://doi.org/10.64898/2026.03.19.712895).

A consensus sequence averages over the viral population in a host, so
within-host evolution lives below it, in minor variants. A mutation seen in 2%
of reads may be real or may be an artefact. I wanted to see why a frequency
cutoff is a weak filter, so I simulated 3,000 real variants and 9,000
artefacts with seven per-site features, and compared approaches.

| Approach | ROC AUC | Average precision |
| --- | --- | --- |
| Frequency cutoff at 2% | — | precision 0.45 at recall 0.65 |
| Logistic regression, raw features | 0.984 | 0.950 |
| Logistic regression + distance from centre | 0.999 | 0.997 |
| Gradient-boosted trees, raw features | 0.999 | 0.998 |

**The part I did not expect, and the reason this was worth doing.** My first
model treated strand balance and read position as worthless: removing either
cost no accuracy at all, even though I had built signal into both. They are
symmetric features — artefacts sit at *both* extremes, real variants in the
middle — so their class means are nearly identical (0.501 vs 0.499 for strand
balance) while their mean distances from the centre are not (0.044 vs 0.317). A
linear model cannot represent a U-shape and reports a coefficient near zero.
Encoding `|x − 0.5|` recovers the signal, and so does a non-linear model. I now
look at distributions before trusting a coefficient.

**The number not to believe is the best one.** An AUC of 0.999 means my two
classes are far more separable than anything real: I drew them from distinct
distributions with clean tails and no correlated noise. The *ranking* of the
approaches is what I would defend; the absolute scores are a property of my own
generating process.

**Open questions I could not answer on my own.** Whether these per-site
features are available in the ONS-CIS pipeline output, and how a model trained
on one protocol transfers to another, given that amplicon and capture data
should carry different artefact signatures.

## Running it

```bash
pip install numpy scikit-learn
python bottleneck.py        # ~1 s, writes bottleneck_results.json
python variant_filter.py    # ~1 s, writes variant_filter_results.json
```

Both scripts seed their generators, so the numbers above reproduce exactly.
Requires Python 3.10+, numpy and scikit-learn.

## Files

| File | Contents |
| --- | --- |
| `bottleneck.py` | Within-host Wright–Fisher model, bottleneck sampling, two analytic checks |
| `variant_filter.py` | Simulated variant calls, frequency baseline, three models, ablation |
| `*_results.json` | Numeric output of each run |
