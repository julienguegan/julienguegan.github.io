---
title: "Smart Garden : IoT, ESP32 et Automatisation d'Arrosage"
lang: fr
classes: wide
author_profile: false
layout: splash
read_time: true
---

<nav class="toc" markdown="1">
<header><h4 class="nav__title"><i class="fas fa-file-alt"></i> Sommaire</h4></header>
*  Auto generated table of contents
{:toc .toc__menu}
</nav>

# Introduction : Pourquoi un Jardin Connecté ?

Le jardinage, bien que relaxant, peut rapidement devenir une contrainte lorsqu'il s'agit de maintenir une routine d'arrosage constante, surtout pendant les périodes d'absence ou de forte chaleur. Le projet **Smart Garden** est né de l'envie de mêler l'électronique grand public (IoT) à un besoin concret : l'entretien autonome des plantes d'intérieur.

L'objectif n'est pas seulement d'automatiser, mais aussi de comprendre. Grâce au suivi des données d'humidité, on peut observer comment différentes plantes consomment l'eau selon l'exposition, la température ambiante ou leur stade de croissance. Ce projet utilise un **ESP32**, des capteurs d'humidité capacitifs et une interface mobile via **Blynk** pour offrir une solution complète de suivi et de contrôle.

![Smart Garden Overview](https://github.com/julienguegan/smart_garden/assets/45081450/cda37684-1d80-44c9-aa63-bd82c95ccbbb)

# L'Écosystème Matériel (Hardware)

Le choix des composants est crucial pour la fiabilité du système sur le long terme. Voici le détail de la configuration utilisée.

## 1. Le Cerveau : ESP32

L'ESP32 a été choisi à la place d'un Arduino classique pour plusieurs raisons :

- **Wi-Fi Intégré** : Indispensable pour la connectivité Cloud sans besoin de module supplémentaire (comme l'ESP8266 ou un module Ethernet).
- **Nombreuses Entrées/Sorties** : Idéal pour gérer plusieurs capteurs et pompes simultanément.
- **Performance** : Un processeur dual-core permettant de gérer les communications réseau sans ralentir la logique de contrôle.

## 2. Capteurs d'Humidité du Sol

Nous utilisons des sondes d'humidité analogiques. Elles mesurent la conductivité ou la capacité du sol (selon le modèle). Les modèles capacitifs sont préférés car moins sujets à la corrosion que les modèles résistifs classiques.

## 3. Système de Pompage

Le système utilise des mini-pompes à eau submersibles de 3V-6V DC. Elles sont reliées au réservoir d'eau et permettent d'apporter la quantité exacte nécessaire à chaque plante.

### Schéma de Branchement (Pinout)

| Composant         | Pin ESP32 | Rôle                                      |
| :---------------- | :-------- | :---------------------------------------- |
| **Capteur Sol 1** | GPIO 33   | Lecture Analogique (Humidité)             |
| **Capteur Sol 2** | GPIO 35   | Lecture Analogique (Humidité)             |
| **Pompe 1**       | GPIO 32   | Commande Digitale (via Relais/Transistor) |
| **Pompe 2**       | GPIO 13   | Commande Digitale (via Relais/Transistor) |

# Architecture Logicielle

Le logiciel est développé sous **PlatformIO**, une alternative moderne à l'IDE Arduino facilitant la gestion des dépendances et de la configuration du projet.

## Utilisation de Blynk IoT

[Blynk](https://blynk.io/) sert de pont entre l'ESP32 et l'utilisateur. Il permet de :

- Créer un tableau de bord mobile (sliders, boutons, graphiques).
- Stocker les données dans le cloud.
- Émettre des notifications en cas de réservoir vide ou d'humidité critique.

## Gestion Non-Bloquante

Un aspect crucial du code est l'absence de la fonction `delay()`. Dans un système IoT, `delay()` arrête l'exécution du programme, ce qui déconnecte souvent le module Wi-Fi ou empêche la réception des commandes en temps réel. Nous utilisons à la place des compteurs de temps (`millis()`) et des timers.

### Configuration des Périphériques

Le fichier `main.cpp` commence par définir les identifiants Blynk et les pins matériels.

```cpp
#define BLYNK_TEMPLATE_ID "TMPLo61F7WPj"
#define BLYNK_DEVICE_NAME "smart garden"

#include <WiFi.h>
#include <BlynkSimpleEsp32.h>

// Configuration des Pins
#define SOIL_SENSOR_1_PIN 33
#define PUMP_1_PIN 32

// Variables globales pour le mode auto
int moisture_threshold = 50;
bool switch_auto_mode = false;
```

# Calibration des Capteurs

L'humidité du sol n'est pas une mesure directe. Le capteur renvoie une valeur de tension (entre 0 et 3.3V) que l'ADC de l'ESP32 convertit en un nombre (généralement entre 0 et 4095). Cependant, chaque capteur a ses propres limites.

### Procédure de Calibration

1. **Valeur à l'Air (Sec)** : On note la valeur lue par le capteur lorsqu'il est sec (ex: 3440).
2. **Valeur dans l'Eau (Humide)** : On note la valeur lue lorsqu'il est immergé jusqu'à la limite autorisée (ex: 1555).

Nous utilisons ensuite la fonction `map()` pour convertir ces valeurs en pourcentage :

```cpp
// SENSOR_1_AIR_VALUE = 3440; SENSOR_1_WATER_VALUE = 1555;
int soil_value_1 = analogRead(SOIL_SENSOR_1_PIN);
int soil_percent_1 = map(soil_value_1, SENSOR_1_AIR_VALUE, SENSOR_1_WATER_VALUE, 0, 100);
soil_percent_1 = constrain(soil_percent_1, 0, 100); // Pour éviter les valeurs hors zone
```

# Logique de Contrôle de l'Arrosage

Le système gère deux modes : un mode manuel déclenché par l'application et un mode automatique. L'activation d'une pompe doit être limitée dans le temps pour éviter d'inonder la plante en cas de défaut d'un capteur.

### Fonction d'Activation Temporisée (Non-bloquante)

Voici comment nous activons la pompe pour une durée précise sans geler le reste du programme :

```cpp
unsigned long pump1_startTime = 0;
bool pump1_active = false;
int pump_duration_sec = 2; // Durée d'arrosage par défaut

void activatePump(int pumpPin, int virtualPin, int durationSec) {
  digitalWrite(pumpPin, HIGH); // Allumer la pompe
  pump1_startTime = millis();   // Noter l'heure de début
  pump1_active = true;
  Blynk.virtualWrite(virtualPin, 1); // Mettre à jour l'interface mobile
}

void loop() {
  Blynk.run();
  timer.run();

  // Vérification de la durée d'arrosage
  if (pump1_active && (millis() - pump1_startTime >= (unsigned long)pump_duration_sec * 1000)) {
    digitalWrite(PUMP_1_PIN, LOW); // Éteindre après X secondes
    pump1_active = false;
    Blynk.virtualWrite(V1, 0);     // Mettre à jour l'interface
  }
}
```

### La Lecture Périodique

Au lieu de lire les capteurs dans la boucle `loop()` le plus vite possible (ce qui génère du bruit et consomme du courant), nous utilisons un `BlynkTimer` pour déclencher une mesure toutes les 10 secondes.

```cpp
void myTimerEvent() {
  int moisture = readMoisture(); // Fonction de lecture et mapping
  Blynk.virtualWrite(V0, moisture); // Envoi au cloud

  // Logique Auto
  if (switch_auto_mode && moisture < moisture_threshold && !pump1_active) {
    activatePump(PUMP_1_PIN, V1, pump_duration_sec);
  }
}
```

![Smart Garden components](https://github.com/julienguegan/smart_garden/assets/45081450/b144a38a-7671-4c3b-9780-f8b499abbdba)

# Défis Techniques et Solutions

Durant la conception, plusieurs obstacles ont été rencontrés :

1. **Bruit Électromagnétique** : L'allumage des pompes peut créer des pics de tension (back-EMF) qui font planter l'ESP32.
   - _Solution_ : Ajout de **diodes de roue libre** aux bornes des moteurs et de condensateurs de filtrage sur l'alimentation.
2. **Corrosion des Capteurs** : Les capteurs résistifs bon marché s'oxydent en quelques semaines.
   - _Solution_ : Passage à des capteurs capacitifs où l'électronique est protégée par une couche isolante.
3. **Stabilité Wi-Fi** : Le module peut parfois perdre la connexion.
   - _Solution_ : Implémentation d'une routine de reconnexion automatique dans le code pour éviter que le jardin ne devienne "aveugle".

# Perspectives d'Évolution

Le projet est une base solide qui peut être étendue de multiples façons :

- **Économie d'Énergie** : Utilisation du mode `Deep Sleep` de l'ESP32. Le système se réveillerait toutes les heures, prendrait une mesure, arroserait si nécessaire, puis s'endormirait pour consommer seulement quelques micro-ampères.
- **Ajout de Capteurs** : Température et humidité de l'air (DHT22), luminosité (LDR) pour corréler l'arrosage avec l'évapotranspiration.
- **Tableau de Bord Local** : Serveur web intégré à l'ESP32 pour un contrôle direct sans dépendre du cloud Blynk.

---

> [!TIP]
> Si vous réalisez ce projet, assurez-vous d'utiliser une alimentation séparée ou suffisamment puissante pour les pompes, car l'ESP32 peut redémarrer si la tension chute trop brutalement lors du démarrage d'un moteur.

_Ce projet illustre parfaitement comment quelques lignes de code et quelques composants abordables peuvent transformer la gestion de notre environnement immédiat._
