---
published: false
title: "Séries Temporelles : RNN, LSTM et Transformers"
date: 2025-11-22T08:00:00+01:00
lang: fr
classes: wide
layout: single
categories:
  - blog
tags:
  - RNN
  - LSTM
  - Transformer
  - Attention
  - PyTorch
header:
  teaser: /assets/images/timeseries_header.png
---

Les séries temporelles sont omniprésentes : cours de la bourse, données météorologiques, signaux audio, trafic web, etc. Prédire ou analyser ces séquences de données ordonnées dans le temps est un défi passionnant et complexe. Le Deep Learning a révolutionné ce domaine avec des architectures capables de capturer les dépendances temporelles, qu'elles soient courtes ou longues.

Dans cet article, nous allons explorer deux familles majeures de modèles : les **réseaux récurrents (RNN et LSTM)** et les **Transformers**. Nous commencerons par les bases théoriques, puis nous plongerons dans des exemples pratiques avec **PyTorch**, allant d'une simple sinusoïde à des cas beaucoup plus complexes et réalistes, comme la prédiction de la productivité d'un développeur en fonction de sa consommation de café !

## Partie 1 : Théorie - Des RNN aux Transformers

### Les Réseaux Récurrents (RNN) : Une Mémoire Simple

Contrairement aux réseaux de neurones classiques qui traitent chaque entrée indépendamment, les RNN possèdent une "mémoire". Ils traitent les séquences élément par élément, en conservant un état caché qui contient des informations sur ce qui a été vu précédemment.

La formule de base d'un RNN est :

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

Cependant, les RNN simples souffrent du problème de **disparition du gradient** (vanishing gradient), ce qui les empêche d'apprendre des dépendances sur de longues séquences.

<figure>
  <img src="/assets/images/timeseries_rnn_unrolled.png" alt="Schéma d'un RNN déroulé">
  <figcaption>Un RNN déroulé sur plusieurs pas de temps. L'information circule de gauche à droite.</figcaption>
</figure>

### Les LSTM : Une Mémoire Longue Durée

Pour pallier ce problème, les **Long Short-Term Memory networks (LSTM)** ont été introduits. Grâce à une structure complexe de "portes" (oubli, entrée, sortie), ils peuvent décider explicitement quelles informations garder ou oublier sur de longues périodes.

<figure>
  <img src="/assets/images/timeseries_lstm_cell.png" alt="Schéma d'une cellule LSTM">
  <figcaption>Structure interne d'une cellule LSTM.</figcaption>
</figure>

### Les Transformers : L'Attention est Tout ce dont vous avez Besoin

Les RNN et LSTM traitent les données séquentiellement, ce qui limite la parallélisation. Les **Transformers** (2017) ont changé la donne en utilisant le mécanisme d'**Attention**. Au lieu de compresser le passé dans un vecteur, l'attention permet au modèle de "regarder" directement n'importe quel point du passé pour comprendre le présent.

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

Cela permet une parallélisation massive et une meilleure capture des dépendances à très long terme.

<figure>
  <img src="/assets/images/timeseries_attention_mechanism.png" alt="Schéma du mécanisme d'attention">
  <figcaption>Le mécanisme d'attention Scaled Dot-Product Attention.</figcaption>
</figure>

---

## Partie 2 : Pratique - De la Théorie au Code

Nous allons utiliser **PyTorch** pour comparer ces modèles. Le code complet est disponible sur le [dépôt GitHub](https://github.com/julienguegan/notebooks_blog).

### Niveau 1 : La Base (Sinusoïde)

Commençons par le "Hello World" des séries temporelles : prédire une fonction sinus. C'est simple, propre et sans bruit.

```python
# Génération d'une onde sinusoïdale simple
def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration))
    y = np.sin(2 * np.pi * freq * t)
    return t, y
```

Les modèles LSTM et Transformer apprennent cette tâche sans difficulté.

<figure>
  <img src="/assets/images/timeseries_sine_wave_prediction.png" alt="Prédiction Sinusoïde">
  <figcaption>Sur une série simple, les deux modèles sont parfaits.</figcaption>
</figure>

### Niveau 2 : La Complexité Augmente

La réalité est rarement aussi propre qu'un sinus. Ajoutons du piment avec des harmoniques, des tendances, de la modulation et même du chaos.

#### Cas A : Multi-fréquences et Tendance

Ici, nous combinons plusieurs ondes, ajoutons une tendance linéaire et quadratique, ainsi que du bruit.

```python
def generate_complex_wave_v1(sample_rate, duration):
    # ... (combinaison de sinus, harmoniques et tendances)
    y += 0.02 * t + 0.001 * t**2 # Tendance
    y += noise_level * np.random.randn(len(t)) # Bruit
    return t, y
```

<figure>
  <img src="/assets/images/timeseries_complex_wave_1.png" alt="Prédiction Onde Complexe 1">
  <figcaption>Le LSTM suit bien la tendance, le Transformer capture mieux les pics rapides.</figcaption>
</figure>

#### Cas B : Modulation (AM/FM) et Bursts

Un signal radio ou audio ressemble souvent à ceci : modulation d'amplitude et de fréquence, avec des "bursts" soudains.

<figure>
  <img src="/assets/images/timeseries_complex_wave_2.png" alt="Prédiction Onde Modulée">
  <figcaption>La modulation met à l'épreuve la capacité du modèle à s'adapter à des changements de régime.</figcaption>
</figure>

#### Cas C : Chaos et Saisonnalité

Mélangeons des cycles saisonniers (comme les ventes annuelles) avec une composante chaotique (type attracteur de Lorenz). C'est le cauchemar des modèles linéaires classiques.

<figure>
  <img src="/assets/images/timeseries_complex_wave_3.png" alt="Prédiction Chaos">
  <figcaption>Même avec du chaos déterministe, les modèles de Deep Learning parviennent à anticiper la dynamique.</figcaption>
</figure>

---

## Partie 3 : Cas "Réel" - Café vs Productivité ☕️

Pour finir, prenons un exemple plus... pragmatique. Imaginons que nous voulons prédire la **productivité d'un développeur** (sur une échelle arbitraire) en fonction de plusieurs facteurs :

1.  **Rythme Circadien** : On dort la nuit (productivité basse).
2.  **Caféine** : Des pics de productivité après le café de 8h et 14h (avec un déclin exponentiel).
3.  **Week-end** : Productivité plus faible (ou différente).
4.  **Bugs de Prod** : Des chutes brutales et aléatoires de productivité ("Server Outages").

```python
def generate_coffee_productivity(days=100):
    # ...
    # Pics de caféine
    coffee_effect[idx_8am:idx_8am+5] += np.exp(-np.arange(5)/2) * 2.0

    # Chutes aléatoires (Bugs)
    outages[idx:idx+4] = -2.0

    productivity = 5 + 2 * circadian + coffee_effect + outages
    return t, productivity
```

Nous entraînons nos modèles sur 30 jours de vie de ce développeur simulé. Voici le résultat :

<figure>
  <img src="/assets/images/timeseries_coffee_and_productivity.png" alt="Prédiction Café Productivité">
  <figcaption>Le modèle apprend les cycles de sommeil et les boosts de café, mais ne peut évidemment pas prédire les bugs aléatoires (les chutes brutales) qui n'ont pas de précurseurs dans le passé !</figcaption>
</figure>

C'est un point crucial : **un modèle ne peut prédire l'aléatoire pur**. Il apprend le rythme circadien et l'effet du café (qui sont réguliers), mais échoue logiquement à anticiper les pannes de serveur aléatoires. Cependant, il se réajuste très vite après l'incident.

## Conclusion

Les séries temporelles modernes nécessitent des outils puissants. Si les méthodes statistiques (ARIMA) restent valables pour des cas simples, le Deep Learning (LSTM, Transformers) excelle dès que la complexité, la non-linéarité et la dimensionnalité augmentent.

Pour aller plus loin, on pourrait explorer :

- Les **Transformers temporels spécialisés** (Informer, Autoformer).
- L'ajout de **variables exogènes** (ex: donner au modèle l'heure de la journée ou la quantité de café bue en entrée explicite).

---

[![Generic badge](https://img.shields.io/badge/écrit_avec-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/acces_au_code-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/script/time_series.py)
