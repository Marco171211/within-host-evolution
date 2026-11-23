# Évolution virale intra-hôte : deux petites études

*(Version française de README.md. C'est la version anglaise qui va sur GitHub ;
celle-ci est là pour la lecture.)*

Deux courtes études en Python écrites pour comprendre deux résultats du groupe
de Katrina Lythgoe à Oxford. Ce sont des modèles jouets sur données simulées.
Ni l'une ni l'autre n'apporte quoi que ce soit au domaine : c'est la façon dont
je me suis expliqué à moi-même ce que les articles affirment.

## 1. `bottleneck.py` — pourquoi une mutation nouvelle est rarement transmise

Motivé par :

- Lythgoe et al., *Science* 2021, [doi:10.1126/science.abg0821](https://doi.org/10.1126/science.abg0821)
  — le goulot d'étranglement à la transmission du SARS-CoV-2 est étroit, donc
  la plupart des variants intra-hôte sont perdus à la transmission.
- Ghafari et al., *Nature* 2024, [doi:10.1038/s41586-024-07029-4](https://doi.org/10.1038/s41586-024-07029-4)
  — 0,1 à 0,5 % des infections persistent 60 jours ou plus, avec une forte
  sélection positive chez certaines.

La question que je voulais trancher moi-même : pour qu'une mutation soit
transmise, il faut d'abord qu'elle devienne fréquente chez son hôte. Quelle est
la part de la durée de l'infection, et quelle est la part de la sélection ?

Une mutation nouvelle est suivie dans une population de Wright-Fisher de 5 000
virions, avec un coefficient de sélection `s`, sur 4 000 réplicats, puis à
travers un goulot de `Nb` virions.

**Deux vérifications avant de croire quoi que ce soit.** L'échantillonnage
Monte-Carlo reproduit `P(transmise) = 1 − (1 − f)^Nb` à 0,003 près sur 24 cas.
Et la probabilité qu'une mutation avantageuse soit encore présente après 100
générations correspond à l'approximation classique `2s` de Haldane (0,043 /
0,100 / 0,175 simulés contre 0,04 / 0,10 / 0,20 pour s = 0,02 / 0,05 / 0,10),
en s'écartant quand `s` grandit, comme prévu.

**Ce qui en ressort.** Une mutation neutre a atteint 5 % de fréquence dans 0
réplicat sur 4 000 après 100 générations ; avec `s = 0,10`, dans 17,4 % des
cas. Et la durée compte en plus : à `s = 0,10`, la probabilité de passer un
goulot de 3 passe de 0,004 à 20 générations à 0,170 à 100, soit un facteur 41.
La sélection décide si une mutation peut monter ; la durée décide jusqu'où elle
monte avant la transmission.

**Limites.** Sélection nulle ou constante, une seule mutation à la fois, pas de
recombinaison, pas de dynamique immunitaire, taille de population constante,
pas d'épistasie. Les vraies populations intra-hôte violent tout cela. Ajouter
une sélection variable dans le temps, pilotée par l'immunité, serait l'étape
suivante évidente.

## 2. `variant_filter.py` — vrais variants minoritaires contre artéfacts

Motivé par Hamilton et al. (preprint 2026), *Identification and Masking of
Artefactual and Misleading Within-Host Variants in Deep-Sequencing SARS-CoV-2
Data*, [doi:10.64898/2026.03.19.712895](https://doi.org/10.64898/2026.03.19.712895).

Une séquence consensus moyenne la population virale d'un hôte : l'évolution
intra-hôte se joue donc en dessous, parmi les variants minoritaires. L'ennui
pratique, c'est qu'une mutation vue dans 2 % des lectures peut être réelle ou
être un artéfact. Je voulais voir pourquoi un seuil de fréquence est un filtre
faible, j'ai donc simulé 3 000 vrais variants et 9 000 artéfacts avec sept
variables par site, et comparé les approches.

| Approche | AUC ROC | Précision moyenne |
| --- | --- | --- |
| Seuil de fréquence à 2 % | — | précision 0,45 à rappel 0,65 |
| Régression logistique, variables brutes | 0,984 | 0,950 |
| Régression logistique + distance au centre | 0,999 | 0,997 |
| Arbres de gradient, variables brutes | 0,999 | 0,998 |

**La partie que je n'attendais pas, et la raison pour laquelle ça valait le
coup.** Mon premier modèle traitait le biais de brin et la position dans la
lecture comme inutiles : retirer l'un ou l'autre ne coûtait aucune performance,
alors que j'avais mis du signal dans les deux. Ces variables sont symétriques —
les artéfacts sont aux *deux* extrêmes, les vrais variants au milieu — donc
leurs moyennes par classe sont presque identiques (0,501 contre 0,499 pour le
biais de brin) tandis que leurs distances moyennes au centre ne le sont pas du
tout (0,044 contre 0,317). Un modèle linéaire ne sait pas représenter une forme
en U et renvoie un coefficient proche de zéro. Encoder `|x − 0,5|` récupère le
signal, et un modèle non linéaire aussi. Je regarde désormais les distributions
avant de faire confiance à un coefficient.

**Le chiffre à ne pas croire est le meilleur.** Une AUC de 0,999 signifie que
mes deux classes sont bien plus séparables que la réalité : je les ai tirées de
lois distinctes, avec des queues propres et sans bruit corrélé. C'est le
*classement* des approches que je défendrais ; les scores absolus sont une
propriété de mon propre générateur de données.

**Questions ouvertes que je n'ai pas pu trancher seul.** Ces variables par site
existent-elles dans la sortie d'un vrai pipeline de surveillance ? Et comment
un modèle entraîné sur un protocole se transfère-t-il à un autre, puisque les
données d'amplicons et de capture devraient porter des signatures d'artéfacts
différentes ?

## Faire tourner

```bash
pip install numpy scikit-learn
python bottleneck.py        # ~1 s, écrit bottleneck_results.json
python variant_filter.py    # ~1 s, écrit variant_filter_results.json
```

Les deux scripts fixent leurs graines aléatoires, donc les chiffres ci-dessus
se reproduisent à l'identique. Nécessite Python 3.10 ou plus, numpy et
scikit-learn.

## Les fichiers

| Fichier | Contenu |
| --- | --- |
| `bottleneck.py` | Modèle de Wright-Fisher intra-hôte, tirage au goulot, deux vérifications analytiques |
| `variant_filter.py` | Appels de variants simulés, référence par seuil, trois modèles, ablation |
| `bottleneck_fr.py`, `variant_filter_fr.py` | Les mêmes, commentaires et affichages en français |
| `*_results.json` | Sortie numérique de chaque exécution |
