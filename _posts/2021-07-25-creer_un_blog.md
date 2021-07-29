---
title: "Créer un blog Jekyll : Ruby, Markdown, Github, Latex"
date: 2019-04-18T15:34:30-04:00
classes: wide
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
---

J'écris le 1er post de ce blog pour parler tout simplement de comment créer un blog comme celui-ci. La raison principale qui m'a poussé à utiliser Jekyll pour créer un blog est la possibilité d'écrire facilement des équations $$\LaTeX$$. En effet, j'avais auparavant essayé avec WordPress mais aucune des solutions que j'avais pu testé m'ont véritablement convaincu. En faisant donc quelques recherches sur le sujet, je suis tombé sur [Jekyll](https://jekyllrb.com/) qui semble être utilisé par un bon nombre de blogger scientifique et informatique. Jekyll est un générateur de site statique c'est-à-dire que les pages web créées ne change pas en fonction de l'internaute qui les visite : tous le monde voit le même contenu. A l'inverse d'un site dynamique qui génère son contenu selon des caractéristiques de la demande (heure, adresse IP, compte utilisateur, formulaire ...). Jekyll est donc une solution tout à fait adaptée pour l'écriture d'un blog web, mais sachez tout de même que ce n'est pas l'unique solution sur le sujet puisque [Hugo](https://gohugo.io/) est également un Framework populaire similaire à Jekyll.

## Installation

Pour utiliser Jekyll sous Windows (comme moi), je vous conseille dans un 1er temps d'installer le [Sous-système Windows pour Linux](https://docs.microsoft.com/fr-fr/windows/wsl/about) qui permet de profiter d'un environnement Linux. Pour ce faire, vous pouvez simplement télécharger l'application WSL en passant par le Microsoft Store.

![alt text](/assets/images/windows_subsytem_linux.png)

Ensuite, il faut installer **Ruby** qui est le langage de programmation utilisé par Jekyll pour fonctionner. En ouvrant l'application Ubuntu précédemment téléchargée et en rentrant les commandes suivantes les unes après les autres dans la console, Ruby devrait être présent sur votre ordinateur:

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-add-repository ppa:brightbox/ruby-ng
sudo apt-get update
sudo apt-get install ruby2.5 ruby2.5-dev build-essential dh-autoreconf
```

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

**Attention:** Votre répertoire git doit être de visibilité public
{: .notice--warning}

## Utilisation

template starter pack minimal mistakes. katex
_config.yml - posts - templates

