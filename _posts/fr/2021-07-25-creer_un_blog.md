---
title: "Créer un blog avec Jekyll : Markdown, Github, Latex"
date: 2021-07-25T15:34:30-04:00
lang: fr
classes: wide
layout: single
categories:
  - blog
tags:
  - Jekyll
  - Ruby
  - Markdown
  - Kramdown
  - Github
  - Latex
  - Katex
  - Blog
header:
  teaser: /assets/images/teaser_blog_jekyll.png
---

J'écris le 1er post de ce blog pour parler tout simplement de comment créer un blog comme celui-ci. La raison principale qui m'a poussé à utiliser Jekyll pour créer un blog est la possibilité d'écrire facilement des équations $\LaTeX$. En effet, j'avais auparavant essayé avec WordPress mais aucune des solutions que j'avais pu testé m'ont véritablement convaincu. En faisant donc quelques recherches sur le sujet, je suis tombé sur [Jekyll](https://jekyllrb.com/) qui semble être utilisé par un bon nombre de blogger scientifique et informatique. Jekyll est un générateur de site statique c'est-à-dire que les pages web créées ne changent pas en fonction de l'internaute qui les visite : tout le monde voit le même contenu. A l'inverse d'un site dynamique qui génère son contenu selon des caractéristiques de la demande (heure, adresse IP, compte utilisateur, formulaire ...). De plus, Jekyll permet d'éditer du texte en **Markdown** en se basant sur la librairie *Kramdown* qui convertie automatiquement du texte Markdown en HTML. Jekyll est donc une solution tout à fait adaptée pour l'écriture d'un blog web scientifique, mais sachez tout de même que ce n'est pas l'unique solution sur le sujet puisque [Hugo](https://gohugo.io/) est également un Framework populaire similaire à Jekyll.

## Installation

Pour utiliser Jekyll sous Windows, une façon de faire est de passer par le [Sous-système Windows pour Linux](https://docs.microsoft.com/fr-fr/windows/wsl/about) qui permet de profiter d'un environnement Linux. Pour ce faire, vous pouvez simplement télécharger l'application WSL en passant par le Microsoft Store.

<p align="center">
   <img src="/assets/images/windows_subsytem_linux.png" width="60%"/>
</p>

Ensuite, il faut installer **Ruby** qui est le langage de programmation utilisé par Jekyll pour fonctionner. En ouvrant l'application Ubuntu précédemment téléchargée et en rentrant les commandes suivantes les unes après les autres dans la console, Ruby devrait être présent sur votre ordinateur:

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-add-repository ppa:brightbox/ruby-ng
sudo apt-get update
sudo apt-get install ruby2.5 ruby2.5-dev build-essential dh-autoreconf
```

**Note:** Pour ceux qui ne veulent pas passer par le sous-système Linux, vous pouvez plus simplement télécharger l'installateur de Ruby ici : [https://rubyinstaller.org/downloads/](https://rubyinstaller.org/downloads/). Vous pourrez ensuite lancer les commandes Jekyll qui suivent dans une fenêtre d'invite de commande Windows.
{: .notice--primary}

Finalement, mettez à jour la commande **gem** et installez **Jekyll** :

```bash
gem update
gem install jekyll bundler
```

## Déploiement

Vous avez désormais tous les pré-requis minimums pour créer un blog avec Jekyll. Commencez par créer un répertoire où vos fichiers de blog seront stockés, créez votre blog avec Jekyll puis construisez le site et rendez le disponible sur un serveur local :

```bash
mkdir mon_blog
jekyll new mon_blog
cd mon_blog
bundle exec jekyll serve
```

Pour naviguer sur votre site en local, rendez-vous sur l'adresse [http://localhost:4000](http://localhost:4000).

**Astuce:** Utilisez l'option ```--livereload``` pour rafraîchir automatiquement la page à chaque changement que vous faites dans les fichiers sources : ```bundle exec jekyll serve --livereload```
{: .notice--info}

Vous avez maintenant générer un site statique avec Jekyll : Bravo, vous pouvez être fier de vous ! Mais je suppose que ça ne vous suffit pas, vous voulez également le rendre disponible à tous le monde. L'une des manières de faire est de l'héberger sur **Github**. En effet, Jekyll a été développé par le fondateur de Github et le déploiement d'un site est possible en utilisant l'outil [Github-Pages](https://pages.github.com/). Il suffit de créer un répertoire git ayant pour nom ```<username_github>.github.io```, générer votre site Jekyll dans ce répertoire et pousser ce répertoire sur Github. Votre blog ```<username_github>.github.io``` sera désormais visible depuis n'importe quel web explorer.

**Attention:** Votre répertoire git doit être de visibilité public. Pour plus d'informations sur le déploiement avec Github, visitez ce [site](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll)
{: .notice--warning}

## Utilisation

Maintenant que vous avez généré votre site, Jekyll a normalement dû créer automatiquement des sous-répertoires et des fichiers dans votre répertoire principal. C'est en modifiant ces fichiers que vous pourrez configurer et personnaliser votre site. Ci-dessous, quelques précisions sur la fonction de certains de ces répertoires et fichiers :

| FICHIER/REPERTOIRE |      DESCRIPTION      |
|--------------------|-----------------------|
| **_config.yml**    | Stocke les données de configuration. Modifier ici le nom du site, les informations de l'auteurs, les plugins, les extensions ...|
| **_includes**      | Les fichiers externes qui permettent d'ajouter des fonctionnalités et être utiliser par les fichiers de Template. |
| **_layouts**       | Les modèles d'affichage qui enveloppent les posts du blog. On peut choisir des mises en page différentes pour chaque post. |
| **_posts**         | Le contenu de votre blog, c'est-à-dire les posts que vous allez écrire. Leurs noms doivent suivre le format : `YEAR-MONTH-DAY-title.MARKUP`|
| **_data**          | Les données externes automatiquement chargées et qui sont utilisées par votre site doivent être stockées ici. |

Par défaut, Jekyll génère le site avec le thème [minima](https://github.com/jekyll/minima) qui permet d'avoir une version simple et épuré mais il existe un grand nombre de [templates](http://jekyllthemes.org/) qui vous permet de personnaliser l'apparence de votre site. Pour ma part, j'ai choisi d'utiliser [minimal-mistakes](https://mmistakes.github.io/minimal-mistakes/) qui est assez simple tout en offrant un grand nombre de possibilité.

**Exemple:** Minimal Mistakes met à disposition un [starter](https://github.com/mmistakes/mm-github-pages-starter/generate) qui permet de rapidement et automatiquement mettre en place les fichiers sur votre compte Github et avoir un site hébergé par Github Pages.
{: .notice--info}

Une fonctionnalité qui m'intéressait particulièrement pour mon blog est de pouvoir facilement ajouter des équations. Le langage le plus connu pour écrire des mathématiques est **Latex** qui est généralement utiliser avec son compilateur pour générer des pdf. Pour le web, la bibliothèque populaire **MathJax** écrite en Javascript est capable d'afficher des équations Latex sur la plupart des navigateurs web courant. Cependant, j'ai préféré choisir la librairie **Katex** qui a l'avantage d'être plus rapide à charger que MathJax quand il y a beaucoup d'équation à convertir (voir exemple ci-dessous, Katex à gauche et MathJax à droite).

<p align="center">
   <img src="/assets/images/katex_vs_latex.gif" width="100%"/>
</p>

Pour installer Katex sur votre blog, il faut copier/coller les lignes suivantes dans le fichier *head.html*, elle permettent de charger la librairie Katex sur votre site :

```javascript
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.css" integrity="sha384-RZU/ijkSsFbcmivfdRBQDtwuwVqK7GMOw6IMvKyeWL2K5UAlyp6WonmB8m7Jd0Hn" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/katex.min.js" integrity="sha384-pK1WpvzWVBQiP0/GjnvRxV4mOb0oxFuyRxJlk6vVw146n3egcN5C925NCP7a7BY8" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.13/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
```

Voilà, c'est tout pour moi. Il y a sûrement des détails à ajouter et des explication supplémentaires à donner, vous pouvez trouver plus d'informations dans les sites ci-dessous et vous pouvez également me poser des questions dans la section commentaire.

## References

- [jekyll](https://jekyllrb.com/)
- [github-pages](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/creating-a-github-pages-site-with-jekyll)
- [minimal-mistakes](https://mmistakes.github.io/minimal-mistakes/)
- [katex](https://katex.org/docs/autorender.html)
- [kramdown](https://kramdown.gettalong.org/index.html)
