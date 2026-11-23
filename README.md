# Why a new mutation almost never gets transmitted

A small simulation I wrote to check whether I had understood two papers on
within-host viral evolution. It is a toy model, not a contribution to the
field: every number it produces is a property of the model rather than a
measurement of anything real.

I am a first-year engineering student at Institut Agro Rennes-Angers, on the
statistics track, taking the MITx MicroMasters in Statistics and Data Science
alongside my degree. Written in November 2026.

## The question

Two results set it up:

- Lythgoe et al., *Science* 2021, [doi:10.1126/science.abg0821](https://doi.org/10.1126/science.abg0821).
  When the virus passes from one person to the next, only a handful of
  particles get through. Most of the variation inside a person is lost at that
  moment.
- Ghafari et al., *Nature* 2024, [doi:10.1038/s41586-024-07029-4](https://doi.org/10.1038/s41586-024-07029-4).
  0.1 to 0.5% of infections last 60 days or more, with strong selection in some
  of them and almost no change in others.

Reading them together left me with a question. For a mutation to be passed on,
it first has to become common inside the person carrying it. So what does the
work: the length of the infection, or selection?

## What the script does

**Part 1.** A mutation present in a fraction `f` of a person's virus particles
gets through a transmission of `n` particles with probability `1 - (1 - f)^n`.
I check that random sampling agrees with the formula before relying on it: over
24 combinations, the largest gap was 0.0025, which is sampling noise.

At 1% frequency, even with 10 particles transmitted, the mutation gets through
less than once in ten transmissions. So it has to become common first.

**Part 2.** I follow one new mutation, starting in a single particle out of
5,000, through an infection, and repeat that 4,000 times for each setting. Each
generation the whole population is replaced: every new particle picks a parent
at random. That random picking is genetic drift — nothing else has to be added
for it to appear. A mutation with an advantage gets a small push each
generation before the copying step.

**Part 3.** The check that matters. Haldane showed in 1927 that a mutation with
advantage `s`, starting in one individual, survives with probability close to
`2s` when `s` is small. My simulation gives 0.043, 0.100 and 0.175 for
`s` = 2%, 5% and 10%, against 0.04, 0.10 and 0.20. It matches for small
advantages and drifts apart for larger ones, exactly as the approximation
should.

## What comes out

With no advantage, the mutation reached 5% frequency in **0 infections out of
4,000** after 100 generations. Luck alone almost never makes a mutation common.

With a 10% advantage, its chance of being transmitted grows from **0.41%** at
20 generations to **16.96%** at 100, about 41 times higher. The no-advantage
case stays flat however long you wait.

So the two do different jobs. Selection decides whether a mutation can rise at
all; time decides how far it rises before transmission. A short infection
exports almost nothing new, and a long one matters only when something is
pushing a mutation upward.

One number worth sitting with: a mutation with a 5% advantage still dies out
90% of the time. Being useful is not enough.

## What the model leaves out

A constant advantage rather than one that changes as the immune response
changes. One mutation at a time. A population of 5,000, far smaller than a real
infection. No recombination. Real infections break all of these assumptions,
and immune-driven selection that varies over time is the first thing I would
add.

## Running it

```bash
pip install numpy
python bottleneck.py
```

Runs in under a second. The random seed is fixed, so the numbers above
reproduce exactly. Requires Python 3.10 or later and numpy.

| File | Contents |
| --- | --- |
| `bottleneck.py` | The simulation |
| `bottleneck_fr.py` | The same code, comments and output in French |
| `LISEZMOI.md` | This write-up in French |
