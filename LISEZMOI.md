# Pourquoi une mutation nouvelle n'est presque jamais transmise

*(Version française de README.md. C'est la version anglaise qui sert de page
d'accueil au dépôt.)*

Une petite simulation écrite pour vérifier si j'avais compris deux articles sur
l'évolution virale intra-hôte. C'est un modèle jouet, pas un apport au
domaine : chaque nombre qu'il produit est une propriété du modèle, pas une
mesure de quoi que ce soit de réel.

Je suis élève ingénieur en première année à l'Institut Agro Rennes-Angers,
parcours statistiques, et je suis le MicroMasters du MIT en statistiques et
science des données en parallèle. Écrit en novembre 2026.

## La question

Deux résultats posent le décor :

- Lythgoe et al., *Science* 2021, [doi:10.1126/science.abg0821](https://doi.org/10.1126/science.abg0821).
  Quand le virus passe d'une personne à la suivante, seules quelques particules
  passent. L'essentiel de la diversité présente chez une personne est perdu à
  ce moment-là.
- Ghafari et al., *Nature* 2024, [doi:10.1038/s41586-024-07029-4](https://doi.org/10.1038/s41586-024-07029-4).
  0,1 à 0,5 % des infections durent 60 jours ou plus, avec une forte sélection
  chez certaines et presque aucun changement chez d'autres.

Lire les deux ensemble m'a laissé une question. Pour qu'une mutation soit
transmise, il faut d'abord qu'elle devienne fréquente chez la personne qui la
porte. Qu'est-ce qui fait le travail : la durée de l'infection, ou la
sélection ?

## Ce que fait le script

**Partie 1.** Une mutation présente dans une fraction `f` des particules
virales d'une personne passe une transmission de `n` particules avec la
probabilité `1 - (1 - f)^n`. Je vérifie qu'un tirage au hasard est d'accord
avec la formule avant de m'appuyer dessus : sur 24 combinaisons, l'écart
maximal était de 0,0025, soit du bruit d'échantillonnage.

À 1 % de fréquence, même avec 10 particules transmises, la mutation ne passe
pas une fois sur dix. Elle doit donc devenir fréquente d'abord.

**Partie 2.** Je suis une mutation nouvelle, partant d'une seule particule sur
5 000, au fil d'une infection, et je répète 4 000 fois pour chaque réglage. À
chaque génération, toute la population est remplacée : chaque nouvelle
particule choisit un parent au hasard. Ce tirage au hasard est la dérive
génétique — rien d'autre n'est nécessaire pour qu'elle apparaisse. Une mutation
avantageuse reçoit une petite poussée à chaque génération, avant l'étape de
copie.

**Partie 3.** La vérification qui compte. Haldane a montré en 1927 qu'une
mutation d'avantage `s`, partant d'un seul individu, survit avec une
probabilité proche de `2s` quand `s` est petit. Ma simulation donne 0,043,
0,100 et 0,175 pour `s` = 2 %, 5 % et 10 %, contre 0,04, 0,10 et 0,20 attendus.
Ça correspond pour les petits avantages et s'écarte pour les plus grands,
exactement comme l'approximation est censée se comporter.

## Ce qui en sort

Sans avantage, la mutation a atteint 5 % de fréquence dans **0 infection sur
4 000** après 100 générations. La chance seule ne rend presque jamais une
mutation fréquente.

Avec 10 % d'avantage, sa probabilité d'être transmise passe de **0,41 %** à 20
générations à **16,96 %** à 100, soit environ 41 fois plus. Le cas sans
avantage reste plat quelle que soit la durée d'attente.

Les deux ne font donc pas le même travail. La sélection décide si une mutation
peut monter ; la durée décide jusqu'où elle monte avant la transmission. Une
infection courte n'exporte presque rien de nouveau, et une infection longue ne
compte que si quelque chose pousse une mutation vers le haut.

Un chiffre sur lequel il vaut la peine de s'arrêter : une mutation avec 5 %
d'avantage disparaît quand même 90 % du temps. Être utile ne suffit pas.

## Ce que le modèle laisse de côté

Un avantage constant, au lieu d'un avantage qui change avec la réponse
immunitaire. Une seule mutation à la fois. Une population de 5 000, bien plus
petite qu'une vraie infection. Pas de recombinaison. Les vraies infections
brisent toutes ces hypothèses, et une sélection pilotée par l'immunité et
variable dans le temps serait la première chose à ajouter.

## Faire tourner

```bash
pip install numpy
python bottleneck.py
```

Tourne en moins d'une seconde. La graine aléatoire est fixée, donc les chiffres
ci-dessus se reproduisent à l'identique. Nécessite Python 3.10 ou plus récent,
et numpy.

| Fichier | Contenu |
| --- | --- |
| `bottleneck.py` | La simulation |
| `bottleneck_fr.py` | Le même code, commentaires et affichages en français |
| `README.md` | Ce texte en anglais |
