---
title: "Modèles de Deep Learning pour les Séries Temporelles : RNN, LSTM et Transformers"
date: 2025-06-01
categories:
  - Deep Learning
  - Séries Temporelles
tags:
  - RNN
  - LSTM
  - Transformer
  - Attention
  - Python
  - PyTorch
---

Les séries temporelles sont partout : cours de la bourse, données météorologiques, signaux audio, etc. Prédire ou analyser ces séquences de données ordonnées dans le temps est un défi passionnant. Le Deep Learning a révolutionné ce domaine avec des architectures capables de capturer les dépendances complexes au fil du temps. Dans cet article, nous allons explorer deux familles majeures de modèles pour les séries temporelles : les réseaux récurrents (RNN et LSTM) et les Transformers avec leur mécanisme d'attention.

## Partie 1 : Les Réseaux Récurrents (RNN) et les LSTM

Imaginez que vous lisez un livre. Pour comprendre la phrase actuelle, vous avez besoin de vous souvenir de ce qui a été dit précédemment. Les réseaux de neurones classiques (comme les perceptrons multi-couches) traitent chaque entrée indépendamment, sans mémoire des entrées passées. C'est là qu'interviennent les réseaux de neurones récurrents (RNN).

### Les RNN : Une Mémoire Simple

Les RNN sont conçus pour traiter des séquences. Ils ont une "boucle" qui leur permet de conserver une information d'une étape à l'autre. À chaque pas de temps $t$, un RNN prend l'entrée actuelle $x_t$ et l'état caché précédent $h_{t-1}$ pour produire une sortie $y_t$ et un nouvel état caché $h_t$.

La formule de base d'un RNN est la suivante :

$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$
$y_t = g(W_{hy} h_t + b_y)$

Où :
- $x_t$ est l'entrée au temps $t$.
- $h_t$ est l'état caché au temps $t$.
- $y_t$ est la sortie au temps $t$.
- $W_{hh}$, $W_{xh}$, $W_{hy}$ sont les matrices de poids.
- $b_h$, $b_y$ sont les biais.
- $f$ et $g$ sont des fonctions d'activation (comme tanh ou ReLU).

{% include figure.html image_path="/assets/images/rnn_unrolled.png" alt="Schéma d'un RNN déroulé" caption="Un RNN déroulé sur plusieurs pas de temps. L'information circule de gauche à droite." %}

Le problème des RNN simples est qu'ils ont du mal à apprendre les dépendances à long terme. Lors de l'entraînement, le gradient peut devenir très petit (vanishing gradient) ou très grand (exploding gradient) lorsqu'il est propagé à travers de nombreuses étapes de temps.

### Les LSTM : Une Mémoire Améliorée

Pour pallier les limitations des RNN, les Long Short-Term Memory networks (LSTM) ont été introduits. Les LSTM ont une structure interne plus complexe appelée "cellule de mémoire" qui peut stocker et réguler l'information sur de longues périodes.

Une cellule LSTM possède plusieurs portes qui contrôlent le flux d'information :
- **Porte d'oubli (Forget Gate):** Décide quelles informations de l'état de la cellule précédente doivent être oubliées.
- **Porte d'entrée (Input Gate):** Décide quelles nouvelles informations doivent être stockées dans l'état de la cellule.
- **Porte de sortie (Output Gate):** Décide quelle valeur de l'état de la cellule sera sortie.

Ces portes sont implémentées à l'aide de couches sigmoïdes et tanh. La structure détaillée est plus complexe que celle d'un RNN simple, mais l'idée clé est cette capacité à gérer explicitement la mémoire.

{% include figure.html image_path="/assets/images/lstm_cell.png" alt="Schéma d'une cellule LSTM" caption="Structure interne d'une cellule LSTM avec ses différentes portes." %}

Les LSTM sont très efficaces pour modéliser des séquences et ont été largement utilisés dans des domaines comme la traduction automatique, la reconnaissance vocale et l'analyse de séries temporelles.

## Partie 2 : Les Transformers et le Mécanisme d'Attention

Bien que puissants, les RNN et LSTM traitent les séquences séquentiellement, un pas de temps après l'autre. Cela peut être lent et limite leur capacité à capturer des dépendances entre des éléments très éloignés dans la séquence. Les Transformers, introduits en 2017, ont changé la donne en s'appuyant entièrement sur un mécanisme appelé "attention".

### Le Mécanisme d'Attention

L'idée fondamentale de l'attention est de permettre au modèle de "peser" l'importance des différentes parties de la séquence d'entrée lorsqu'il traite un élément particulier. Au lieu de compresser toute l'information passée dans un seul état caché (comme les RNN), l'attention permet au modèle d'accéder directement à n'importe quel élément de la séquence d'entrée.

Le mécanisme d'attention le plus courant est l'attention "Scaled Dot-Product Attention". Il prend trois entrées : les Requêtes (Queries, Q), les Clés (Keys, K) et les Valeurs (Values, V).

L'attention est calculée comme suit :

$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$

Où :
- $Q$, $K$, $V$ sont des matrices dérivées des vecteurs d'entrée.
- $d_k$ est la dimension des clés, utilisée pour mettre à l'échelle le produit scalaire.
- $\text{softmax}$ est appliquée ligne par ligne pour obtenir des poids d'attention.

{% include figure.html image_path="/assets/images/attention_mechanism.png" alt="Schéma du mécanisme d'attention" caption="Le mécanisme d'attention Scaled Dot-Product Attention." %}

Le mécanisme d'attention permet au modèle de créer une représentation pondérée de la séquence d'entrée, où les poids sont déterminés par la pertinence de chaque élément par rapport à l'élément actuel traité.

### Les Transformers

Les Transformers utilisent le mécanisme d'attention (souvent sous forme de "Multi-Head Attention" qui combine plusieurs mécanismes d'attention en parallèle) dans une architecture composée d'un encodeur et d'un décodeur. Pour les séries temporelles, on utilise souvent des architectures basées uniquement sur l'encodeur ou adaptées spécifiquement.

L'avantage majeur des Transformers est leur capacité à traiter la séquence en parallèle, ce qui accélère considérablement l'entraînement sur du matériel moderne (GPU, TPU). De plus, le mécanisme d'attention leur permet de capturer efficacement les dépendances à long terme, car la distance entre les éléments dans la séquence n'affecte pas directement la capacité du modèle à y prêter attention.

## Exemple Pratique : Prédire une Série Temporelle Sinusoïdale avec PyTorch

Pour illustrer ces concepts, nous allons créer un exemple simple de prédiction de série temporelle en utilisant une fonction sinusoïdale avec PyTorch. C'est un exemple facile à visualiser et qui permet de bien comprendre comment ces modèles apprennent à capturer la dynamique séquentielle.

Nous allons générer une série temporelle basée sur la fonction sinus, puis utiliser un modèle LSTM et un modèle Transformer simple pour prédire la valeur suivante de la série en se basant sur une fenêtre de valeurs précédentes.

Voici le code Python pour générer les données et construire les modèles :

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- Génération des données ---
def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, int(sample_rate * duration))
    y = np.sin(2 * np.pi * freq * x)
    return y

freq = 1 # Fréquence de la vague
sample_rate = 100 # Nombre de points par seconde
duration = 10 # Durée en secondes
sine_wave = generate_sine_wave(freq, sample_rate, duration)

# --- Préparation des données pour les modèles séquentiels ---
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        feature = data[i:(i + look_back)]
        target = data[i + look_back]
        X.append(feature)
        Y.append(target)
    return np.array(X), np.array(Y)

look_back = 50 # Nombre de pas de temps précédents à considérer
X_np, Y_np = create_dataset(sine_wave, look_back)

# Convertir en tenseurs PyTorch
X = torch.from_numpy(X_np).float()
Y = torch.from_numpy(Y_np).float().unsqueeze(1) # Ajouter une dimension pour la sortie

# Reshape pour LSTM [batch_size, seq_len, input_size]
X_lstm = X.unsqueeze(2)

# --- Modèle LSTM avec PyTorch ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.unsqueeze(1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions

lstm_model = LSTMModel()
loss_function = nn.MSELoss()
optimizer_lstm = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

print("Entraînement du modèle LSTM...")
epochs = 100
for i in range(epochs):
    for seq, labels in zip(X_lstm, Y):
        optimizer_lstm.zero_grad()
        lstm_model.hidden_cell = (torch.zeros(1, 1, lstm_model.hidden_layer_size),
                                torch.zeros(1, 1, lstm_model.hidden_layer_size))

        y_pred = lstm_model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer_lstm.step()

    if i%25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {epochs:3} loss: {single_loss.item():10.8f}')
print("Entraînement LSTM terminé.")

# --- Modèle Transformer simple (Encoder only) avec PyTorch ---
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=50, nhead=2, num_layers=2, output_size=1, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.linear = nn.Linear(d_model, output_size)
        self.input_linear = nn.Linear(input_size, d_model) # Linear layer to match input_size to d_model

    def forward(self, src):
        # Apply linear layer to match input_size to d_model
        src = self.input_linear(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32)) # Scale input
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.linear(output[:, -1, :]) # Take the output of the last time step
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

transformer_model = TransformerModel(input_size=1, d_model=50, nhead=5, num_layers=2) # Adjusted nhead to be a divisor of d_model
optimizer_transformer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

# Reshape for Transformer [batch_size, seq_len, input_size]
X_transformer = X.unsqueeze(2)

print("Entraînement du modèle Transformer...")
epochs = 100
for i in range(epochs):
    for seq, labels in zip(X_transformer, Y):
        optimizer_transformer.zero_grad()
        y_pred = transformer_model(seq.unsqueeze(0)) # Add batch dimension

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer_transformer.step()

    if i%25 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {epochs:3} loss: {single_loss.item():10.8f}')
print("Entraînement Transformer terminé.")

# --- Prédictions et Visualisation ---
lstm_model.eval()
transformer_model.eval()

lstm_predictions = []
transformer_predictions = []

with torch.no_grad():
    for seq in X_lstm:
        lstm_model.hidden_cell = (torch.zeros(1, 1, lstm_model.hidden_layer_size),
                                torch.zeros(1, 1, lstm_model.hidden_layer_size))
        lstm_predictions.append(lstm_model(seq).item())

    for seq in X_transformer:
        transformer_predictions.append(transformer_model(seq.unsqueeze(0)).item())

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(sine_wave, label='Série Temporelle Originale')
# Adjusting the x-axis for predictions to align with the target values
plt.plot(np.arange(look_back, len(sine_wave)), lstm_predictions, label='Prédictions LSTM')
plt.plot(np.arange(look_back, len(sine_wave)), transformer_predictions, label='Prédictions Transformer')
plt.xlabel('Pas de Temps')
plt.ylabel('Amplitude')
plt.title('Prédiction de Série Temporelle Sinusoïdale')
plt.legend()
plt.grid(True)
plt.savefig('assets/images/sine_wave_prediction.png') # Save the figure
plt.close() # Close the plot to free memory

```

## Résultats et Visualisation

Après avoir entraîné les modèles LSTM et Transformer sur notre série temporelle sinusoïdale, nous pouvons visualiser leurs prédictions par rapport à la série originale. Le code ci-dessus génère un graphique qui montre à quel point chaque modèle a réussi à capturer la dynamique de la vague sinusoïdale et à prédire les valeurs futures.

{% include figure.html image_path="/assets/images/sine_wave_prediction.png" alt="Graphique des prédictions LSTM et Transformer sur une série sinusoïdale" caption="Comparaison des prédictions LSTM et Transformer avec la série temporelle sinusoïdale originale." %}

Comme on peut le voir sur le graphique, les deux modèles devraient être capables de prédire avec une bonne précision la suite de la série sinusoïdale, démontrant leur capacité à apprendre les motifs temporels. Pour des séries temporelles plus complexes, les différences entre les modèles (notamment la capacité des Transformers à gérer les dépendances à long terme) deviendraient plus apparentes.

## Conclusion

Les modèles de Deep Learning comme les RNN, les LSTM et les Transformers offrent des approches puissantes pour l'analyse et la prédiction des séries temporelles. Les RNN et LSTM excellent dans la capture des dépendances séquentielles, tandis que les Transformers, grâce à leur mécanisme d'attention, sont particulièrement efficaces pour les dépendances à long terme et permettent une parallélisation accrue de l'entraînement. Le choix du modèle dépendra de la nature spécifique de la série temporelle et des ressources de calcul disponibles.

Cet exemple simple avec une série sinusoïdale donne un aperçu de la manière dont ces modèles fonctionnent. Pour des applications réelles, des prétraitements de données plus sophistiqués, des architectures de modèles plus complexes et des techniques d'entraînement avancées sont souvent nécessaires.