---
title: "Détection de monture de lunettes par méthode d'intelligence artificielle"
lang: fr
classes: wide
author_profile: false
layout: splash #single #
read_time: true
---

<nav class="toc" markdown="1">
<header><h4 class="nav__title"><i class="fas fa-{{ include.icon | default: 'file-alt' }}"></i> {{ include.title | default: site.data.ui-text[site.locale].toc_label }}</h4></header>
*  Auto generated table of contents
{:toc .toc__menu}
</nav>

# Contexte

Le sujet du stage est alors de développer un algorithme qui détecte sur des images 2D les pixels appartenant à la face avant de la monture portée par l'utilisateur en étant robuste le plus possible à tous les différents types de monture qui peuvent exister ainsi que les différentes positions que peut avoir l'utilisateur devant la caméra. On parle ici d'un problème de segmentation, contrairement à la tâche de classification où le but est simplement d'identifier si oui ou non l'objet recherché est présent dans les images. L'idée étant de, pour les versions suivantes du projet, retrouver des mesures 3D à partir des sorties de cet algorithme de détection qui aura segmenté la monture pour différents angles de tête d'une même personne. Une piste proposée pour cette future problématique est de fitter un modèle de courbes paramétriques (courbe de Bézier, descripteurs de Fourier ...) 3D aux différentes observations 2D obtenues.

La détection de lunettes est un problème clé pour la recherche de vision par ordinateur dû à sa relation directe avec les systèmes de reconnaissance faciale. Le problème fut typiquement approché en localisant la zone des yeux et en caractérisant les régions alentours. Une combinaison de techniques morphologiques du traitement d'images fut testée pour décrire la zone des lunettes comme l'utilisation de motif binaire locaux \cite{local_binary_pattern}, la décomposition en ondelette \cite{wavelet}, les histogrammes de gradient orienté \cite{hog} ou encore les caractéristiques pseudo-Haar \cite{haar_feature}. Ces approches demandent une expertise fine du domaine et connaissent de vraies limitations pour obtenir une segmentation complète pour tous les types de montures existants et sous toutes les conditions de positions et d'environnement. Ces dernières années, les approches d'apprentissage profond ont été largement utilisées et ont permis de grande avancées dans le domaine de la vision par ordinateur. Elles ont l'avantage d'être agnostiques aux problèmes à résoudre mais nécessitent en contrepartie une base de données annotées au préalable. Saddam BEKHET \cite{selfie_detection} présente un modèle robuste pour classifier des images avec ou sans lunettes dans des conditions difficiles mais ne segmente pas précisément la monture. Cheng-Han LEE et al. \cite{celebamask} fournissent un jeu de données annotées pour la segmentation de 19 classes du visage (nez, bouche, cheveux ...), il est composé de plus de 30 000 images. On trouve parmi ces 19 classes, la classe "lunette" définie comme l'ensemble des éléments de la lunette (monture, verre, branche) alors que notre problème exige uniquement une segmentation de la face avant de la monture (pas le verre ni les branches). 

L'approche retenue pour le problème de segmentation de la face avant de la monture est celle de l'apprentissage supervisée avec notamment 2 approches. Une première s'appuyant sur un jeu de données générées synthétiquement et permettant d'avoir accès à une grande quantité d'images annotées. Et une deuxième se basant sur un jeu de données réelles ayant été annotées manuellement et plus spécifique au cas d'application mais avec une faible quantité d'image. Un rappel du fonctionnement général des réseaux de neurone sera fait puis plus spécifiquement sur les architectures propres à la segmentation sémantique dans le chapitre \ref{Etat de l'art}. Ensuite, les différentes expériences effectuées seront présentées dans le chapitre \ref{Travaux}. Enfin, les résultats seront discutés et les potentielles améliorations énumérées dans le chapitre \ref{conclusion}.


# État de l'art

## Architectures pour la Segmentation Sémantique

Dans un réseau de neurones spécialisé dans la classification, le type d'information est encodé mais pas sa position précise. En effet, les opérations de pooling effectuent un sous-échantillonnage de l'information et permettent au modèle de comprendre le type d'information présent dans l'image mais la position de l'information est perdue au cours du procédé. Comme expliqué en introduction, dans le contexte du stage, la tâche à effectuer est celle de la segmentation (cf figure \ref{fig:segmentation_semantic}). L'objectif de la segmentation sémantique est d'étiqueter chaque pixel d'une image par sa classe correspondante. Contrairement à la tâche de classification qui consisterait à dire si oui ou non il existe une monture dans l'image, la dimension de sortie est alors bien inférieure à la dimension d'entrée (en général).

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.5]{Figures/semantic segmentation.png}
    \caption{Différentes tâches de reconnaissance d'image. La segmentation d'instance est la plus complexe.}
    \label{fig:segmentation_semantic}
\end{figure}

Pour la segmentation, la dimension de sortie est quasiment identique à celle d'entrée puisqu'elle correspond exactement à la taille de l'image. Le challenge principal est de retrouver où se trouve l'information dans l'image en effectuant du sur-échantillonnage pour récupérer une sortie de la même taille que l'image d'entrée. Pour se faire, de nombreuses techniques du traitement d'images classique sont connues comme l'interpolation bilinéaire, l'interpolation aux plus proches voisins, le unpooling, la convolution transposée (ou déconvolution)... Comme vous vous en doutez, l'opération retenue pour un réseau de neurones convolutionnel est souvent la convolution transposée puisqu'on va pouvoir apprendre les paramètres grâce à nos données et avoir un sur-échantillonnage adapté au problème. L'idée est de réaliser l'opération inverse de la convolution : en partant du produit et du filtre, retrouver l'image d'origine (cf. figure \ref{fig:convolution_transposed}).

\begin{figure}[H]
    \centering
    \begin{subfigure}[t]{0.35\textwidth}
        \centering
        \includegraphics[scale=0.5]{Figures/convolution.png}
        \caption{Convolution.}
    \end{subfigure}
    
    \begin{subfigure}[t]{0.75\textwidth}
        \centering
        \includegraphics[scale=0.6]{Figures/transposed_convolution.png}
        \caption{Convolution transposée.}
    \end{subfigure}
    \caption{Exemples de calculs de convolution classique et transposée.}
    \label{fig:convolution_transposed}
\end{figure}

Une majorité des architectures des réseaux de neurones pour la segmentation sont constituées d'une partie encodeur qui encode l'information avec des opérations de convolution et de pooling puis d'une partie décodeur qui décode l'information spatialement en utilisant des opérations de déconvolution et de unpooling (cf. figure \ref{fig:deconvolution neural network}). Au final, on obtient un réseau de neurones constitué uniquement de couches convolutionnelles et ne contenant aucune couche dense, ce qui permet d'accepter en entrée des images de n'importe quelle taille. On peut, par exemple, tester des images de taille différentes lors du test et lors de l'entraînement. Dans les sections suivantes, des architectures spécifiques et reconnues seront citées pour mieux comprendre les enjeux rencontrés lors des problèmes de segmentation d'image.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.5]{Figures/deconvolution neural network.png}
    \caption{Architecture complètement convolutionnelle (FCN) pour la segmentation}
    \label{fig:deconvolution neural network}
\end{figure}

### UNet

L'architecture UNet \cite{unet} a été développée par O. Ronneberger en 2014 pour de la segmentation d'images biomédicales. Comme discuté précédemment pour les architectures de segmentation, il y a deux parties principales : l'encodeur qui sous-échantillonne l'information et permet de savoir ce qui est dans l'image et le décodeur qui sur-échantillonne et permet de savoir où est l'information dans l'image. Le principal ajout par rapport à ce qui a été précédemment énoncé est le fait de connecter les couches de même niveau de résolution de l'encodeur et du décodeur. En effet, en concaténant les 2 informations puis en appliquant 2 opérations de convolution, le modèle a une information supplémentaire et peut apprendre comment sur-échantillonner de façon plus efficace. 

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.5]{Figures/unet.png}
    \caption{Architecture UNet}
\end{figure}

### PSPNet

L'architecture PSPNet (Pyramid Scene Parsing) \cite{pspnet} développée en 2016 par H. ZHao et al. de CUHK et l'entreprise Sensetime a réalisé des performances "state-of-the-art" sur les benchmark d'*ImageNet 2016 scene parsing challenge}, *PASCAL VOC 2012} et *Cityscapes}. En observant les mauvaises prédictions faites par un CNN, ils ont conclu que le modèle avait besoin de plus d'information globale de l'image. En effet, il remarque par exemple que le modèle prédit un bateau sur l'eau comme une voiture en se basant sur son apparence alors que le sens commun nous dit qu'une voiture a peu de chance de flotter sur l'eau. 
Pour décrire cette information globale de l'image, l'encodeur de PSPNet utilise des couches de convolutions dilatées qui aident à augmenter le champ récepteur des features. Ensuite des features de différentes tailles sont regroupées par des opérations de pooling. Ces  différentes échelles de contexte sont directement sur-échantillonnées avec des interpolations bilinéaires pour être concaténées et formées un seul volume encodant différents contextes d'information. Ce volume est finalement passées à une couche de convolution pour générer la prédiction finale (cf figure \ref{fig:pspnet}). 

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.4]{Figures/pspnet.png}
    \caption{Architecture PSPNet}
    \label{fig:pspnet}
\end{figure}


### DeepLab

DeepLab \cite{deeplab} est une architecture développée la même année que PSPNet et réalise également des performances du même ordre de grandeur puisque l'idée et les techniques utilisées sont semblables. Elle est notamment connue pour introduire précisément la notion des convolutions dilatées (ou atrous convolution) pour aider à encoder l'information locale et globale de l'image. D'autre part, DeepLab a subi plusieurs améliorations depuis sa création et 3 versions existent actuellement. Les convolutions dilatées (\ref{fig:atrous_convolution}) permettent d'élargir le champ récepteur des filtres et donc de se passer de toutes les couches de déconvolution pour la partie décodage utilisées dans les réseaux complètement convolutionnels (UNet, FCN ...) mais DeepLab V1 est contraint de faire passer la sortie des convolutions dilatées dans une interpolation bilinéaire et un modèle CRF.

\begin{figure}[H]
    \centering
    \captionsetup{width=0.7\textwidth}
    \includegraphics[scale=0.4]{Figures/atrous convolution.png}
    \caption{A gauche, convolution standard. A droite, convolution dilatée (avec un taux de dilatation de 2).}
    \label{fig:atrous_convolution}
\end{figure}

L'idée de DeepLab V2 est, de la même manière que PSPNet, d'effectuer un pooling spatial pyramidal dilaté (ASPP) en appliquant de multiples convolutions dilatées avec des taux d'échantillonnage différents (exemple : 4 kernels 3x3 avec des taux de 6, 12, 18 et 24) et de concaténer les sorties en un seul volume, ce qui aide à prendre en compte différentes échelles d'objet. Enfin, DeepLab V3 \cite{deeplab_v3} et V3+ se concentre sur la capture de frontières plus nettes pour la segmentation des objets. En effet, jusqu'ici le modèle utilisait une interpolation pour le sur-échantillonnage et non un décodeur avec des convolutions comme dans UNet. En utilisant une partie décodeur peu profonde qui exploite les convolutions dilatées, le modèle obtient des résultats plus précis et nets autour des objets segmentés.

### BiseNet

BiseNet \cite{bisenet} est une architecture développé en 2018 par une équipe chinoise ayant pour but de trouver un équilibre entre performance et vitesse d'inférence. En effet, pour accélérer un modèle on peut choisir de : réduire la taille de l'entrée (en rognant ou redimensionnant l'image), réduire les channels du réseau ou supprimer les dernières couches mais à chaque fois au détriment de la perte de détails spatiaux. BiseNet propose alors un réseau avec 2 parties : *Spatial Path} (SP) et *Context Path} (CP). Le *Spatial Path} empile 3 couches convolutionnelles pour obtenir une feature map 1/8 et le *Context Path} ajoute une couche de global average pooling à la fin de Xception où le champ récepteur est maximum. Ensuite pour fusionner ces 2 informations et raffiner la prédiction, des modules appelés *Feature Fusion} (FFM) et *Attention Refinement} (ARM) sont respectivement développés (cf. figure \ref{fig:bisenet}).

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.5]{Figures/bisenet - speedup.png}
    \caption{Architecture BiseNet}
    \label{fig:bisenet}
\end{figure}

## Métriques et fonctions Loss

La première métrique qu'on peut évaluer pour la tâche de segmentation est la précision (accuracy) au niveau pixel qui consiste simplement au pourcentage de pixels de l'image qui sont classifiés correctement. Cette métrique peut être pertinente pour des cas où les classes sont distribuées de façon équilibré sur l'image mais lorsque notre problème est déséquilibré et qu'une classe domine largement l'image et une autre occupe seulement une petite portion, le score de précision peut être trompeur puisqu'une prédiction qui donnerait la valeur de la classe dominante à l'image entière aurait un très bon score.

\begin{figure}[H]
    \centering
    %\captionsetup{width=0.7\textwidth}
    \includegraphics[scale=0.6]{Figures/pixel accuracy - bad example.png}
    \caption{Exemple de classe déséquilibré où le score de précision est inefficace. En prédisant l'image entière comme du background, on obtient quand même une précision de 90\%.}
    \label{accuracy}
\end{figure}

Le score IoU\footnote{Intersection Over Union} aussi appelé indice de Jaccard est l'une des métriques les plus utilisée en segmentation sémantique puisqu'elle exprime l'aire de chevauchement entre prédiction et vérité divisé par l'aire d'union des deux. Elle peut être calculée ainsi : $$\text{IoU} = \frac{TP}{TP+FP+FN}$$ 
où $TP$ désigne les vrais positifs, $FP$ les faux positifs et $FN$ les faux négatifs. Si on reprend l'exemple de la figure \ref{accuracy}, on a 90\% de chevauchement pour la classe background mais 0\% pour la classe lunette, on obtient donc au final un score IoU de 45\% ce qui paraît un peu plus approprié pour la prédiction proposée. Pour un problème binaire comme celui de notre problème, on pourra calculer le score IoU seulement de la classe lunette (pour des problèmes multi-classes, on peut également exclure le background). Une grandeur qui revient également souvent pour la segmentation et qui est très proche de l'IoU est le coefficient Dice (ou F1 score) : $$\text{Dice} = \frac{2TP}{2TP+FP+FN}$$
Ces deux métriques sont corrélées positivement, c'est-à-dire que pour une unique donnée si un classifieur A est meilleur qu'un classifieur B sous l'une des métriques alors il sera également meilleur que B sous l'autre métrique. La différence entre les deux vient quand on compare les deux sur un ensemble de données. En général, la métrique IoU pénalise plus fortement les mauvaises segmentation que le coefficient Dice. On peut penser au coefficient Dice comme une métrique qui mesure une performance moyenne alors le score IoU mesure quelque chose qui s'approche plus de la pire performance.

Les problèmes de classification d'images basées sur des CNN sont typiquement entraînés en minimisant la cross-entropie (page \pageref{CE}) qui mesure une affinité entre la probabilité sortie du réseau et le label. La cross-entropie standard a des inconvénients bien connus pour des problèmes où la distribution des classes sont fortement déséquilibrées, il en résulte des entraînements instables et des frontières de décision biaisées envers la classe majoritaire. Pour les tâches de segmentation, les métriques de score évoquées précédemment peuvent être directement utilisé comme fonction de Loss en utilisant $1 - \text{IoU}$ ou $1 - \text{Dice}$ pour bien avoir un problème de minimisation. En effet, comme expliqué précédemment, elles surpassent les performances de la cross-entropie et sont plus robustes aux problèmes déséquilibrés et en plus, elles permettent au réseau non pas de se concentrer sur le résultat de chaque pixel mais sur la forme globale de l'objet de la segmentation. 

# Travaux effectués

L'objectif des travaux est la segmentation de monture de lunettes pour un instrument de mesure. Seule la face avant de la lunette est considérée dans ces travaux puisque les mesures nécessaires pour l'instrument concernent seulement le plan vertical avant de la monture. Les branches ainsi que les verres ne seront donc pas concernés par la segmentation. De plus, le problème est binaire puisqu'il existe 2 classes : la monture et le background. 

Dû à l'absence d'une base de données annotées de qualité suffisante et de taille suffisante, les sujets de l'apprentissage semi-supervisé et des réseaux génératifs (GAN) ont été abordés. Le premier permet d'utiliser un petit jeu de données annotées et d'exploiter un grand nombre de données non annotées, le second permet de générer des images sans utiliser d'annotation. Ces approches ont été abandonnées, d'une part du fait de leur complexité et d'autre part, l'apport en performance qu'elles peuvent procurer est faible comparé à celui d'une base de données annotées de qualité. 

L'approche retenue ici est finalement celle de l'apprentissage supervisé avec les architectures de réseau de neurones convolutionnel propres à la segmentation (cf section \ref{segmentation}). Pour la phase d'apprentissage, deux approches ont été mises en place. La première est d'utiliser des images synthétiques où la monture a été apposée automatiquement sur le visage d'une personne. La deuxième approche est de constituer manuellement une base de données en étiquetant chaque image collectée. Avant de décrire les expériences réalisées, un bref rappel des travaux précédemment testés au sein de l'entreprise sera fait. Les différentes technologies logiciel et matériel seront également énumérées.

## Travaux précédents

La première méthode qui fut testée dans des travaux antérieurs est une approche de traitement d'images plus classique qui utilise la détection des contours d'une image (cf. figure \ref{fig:resultat_pascal_detection}). En s'appuyant sur la librairie \textbf{Dlib} \cite{dlib}, la position des yeux est récupérée. Puis l'idée est de chercher les premiers contours rencontrés dans toutes les directions en partant de la position de chaque oeil. Un inconvénient de cette approche est qu'elle permet en réalité de détecter seulement le bas de la monture. En effet, le haut de la monture est plus compliqué à détecter avec cette technique puisque le sourcil apparaît souvent entre la monture et l'oeil et est compliqué à filtrer. Un autre inconvénient est la robustesse de l'algorithme de détection de contour qui s'appuie notamment sur l'intensité des gradients de l'image et d'un paramètre de seuil, il est difficile d'obtenir le même résultat pour différentes images.

\begin{figure}[H]
    \centering
    \captionsetup{width=0.7\textwidth}
    \begin{subfigure}[t]{0.4\textwidth}
        \centering
        \includegraphics[scale=0.2]{Figures/pascal_contour_2.jpg}
        \caption{Détection de contour}
    \end{subfigure}
    \begin{subfigure}[t]{0.4\textwidth}
        \centering
        \includegraphics[scale=0.22]{Figures/pascal_contour.png}
        \caption{Détection de monture}
    \end{subfigure}
    \caption{Exemple de résultat de détection de monture en utilisant les contours et la position des yeux}
    \label{fig:resultat_pascal_detection}
\end{figure}

La deuxième approche consistait à utiliser un réseau de neurones pour la segmentation. La base de données d'apprentissage a été constituée en utilisant un \textit{Virtual Try On} pour récupérer automatiquement une image de visage avec monture ainsi que l'étiquette associée. Ainsi, 18 000 images ont été générées en utilisant 60 visages fois 300 montures. Pour chaque image, un pré-traitement est effectué : l'image est rognée autour de la zone des yeux (avec \textbf{Dlib}) puis passée en image contour (cf. figure \ref{fig:shuang_process}). L'inconvénient de ce pré-traitement est que la prédiction finale du réseau de neurones dépend aussi de l'efficacité de la détection de contour. De plus, cette détection de contour n'est pas nécessaire puisque le réseau de neurones est capable d'apprendre ce genre de caractéristique dans les premières couches du réseau. Enfin, aucune métrique de score ne fut développée et il a donc été impossible de savoir la véritable performance du modèle entraîné.

\begin{figure}[H]
    \centering
    \includegraphics[scale=0.4]{Figures/shuang_process.png}
    \caption{Traitement effectué sur les images}
    \label{fig:shuang_process}
\end{figure}

Finalement, les 2 méthodes citées précédemment sont peu robustes aux différentes positions de la tête qui est l'un des objectifs principaux de l'étude.

## Expériences sur données synthétiques

### Base de données

Une première partie des travaux fut d'utiliser des données synthétiques pour la base d'apprentissage. En effet, aucune base de données annotées correspondant à la segmentation voulue existe au sein de l'entreprise ou en open-source. Ces données synthétiques sont générées avec le même outil de \textit{Virtual Try On} (VTO) cité en section \ref{previous_work}. Cet outil utilise les landmarks du visage pour y superposer une image de lunette sur la zone des yeux. Avec ce procédé, il est possible de connaître exactement les pixels modifiés et donc d'avoir l'étiquette associée à chaque image nécessaire pour l'apprentissage supervisé (cf. figure \ref{fig:virtual_database}). La base de donnée publique \textit{Labeled Faces in the Wild} \cite{LFW_database} composée de \textasciitilde 10 000 images (et \textasciitilde 5000 personnes différents) a été utilisée. Les images sont de taille 250*250 pixels. Pour chaque image, une monture choisie aléatoirement parmi 50 a été placée avec le VTO. Ainsi, la base de donnée obtenue est assez variée avec beaucoup de visages différents chacun dans des conditions différents (position, luminosité, arrière-plan). Aucun pré-traitement n'est appliqué sur l'image comme ce fut le cas dans les travaux précédents.

\begin{figure}[H]
    \centering
    \captionsetup{width=0.7\textwidth}
    \begin{subfigure}[t]{0.4\textwidth}
        \centering
        \includegraphics[scale=0.3]{Figures/virtual_images.png}
        \caption{Images}
    \end{subfigure}
    \begin{subfigure}[t]{0.4\textwidth}
        \centering
        \includegraphics[scale=0.3]{Figures/virtual_masks.png}
        \caption{Masques}
    \end{subfigure}
    \caption{Base de donnée d'apprentissage générée à partir du VTO}
    \label{fig:virtual_database}
\end{figure}

### Expériences