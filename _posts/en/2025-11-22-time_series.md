---
published: false
title: "Time Series: RNN, LSTM and Transformers"
date: 2025-11-22T08:00:00+01:00
lang: en
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

Time series are everywhere: stock prices, weather data, audio signals, web traffic, etc. Predicting or analyzing these time-ordered data sequences is a fascinating and complex challenge. Deep Learning has revolutionized this field with architectures capable of capturing temporal dependencies, whether short or long.

In this article, we will explore two major families of models: **recurrent networks (RNN and LSTM)** and **Transformers**. We'll start with the theoretical foundations, then dive into practical examples with **PyTorch**, ranging from a simple sine wave to much more complex and realistic cases, such as predicting a developer's productivity based on their coffee consumption!

## Part 1: Theory - From RNN to Transformers

### Recurrent Neural Networks (RNN): A Simple Memory

Unlike classical neural networks that process each input independently, RNNs have a "memory". They process sequences element by element, maintaining a hidden state that contains information about what has been seen previously.

The basic formula of an RNN is:

$$h_t = f(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

However, simple RNNs suffer from the **vanishing gradient** problem, which prevents them from learning dependencies over long sequences.

<figure>
  <img src="/assets/images/timeseries_rnn_unrolled.png" alt="Unrolled RNN diagram">
  <figcaption>An RNN unrolled over several time steps. Information flows from left to right.</figcaption>
</figure>

### LSTM: Long-Term Memory

To address this problem, **Long Short-Term Memory networks (LSTM)** were introduced. Thanks to a complex structure of "gates" (forget, input, output), they can explicitly decide which information to keep or forget over long periods.

<figure>
  <img src="/assets/images/timeseries_lstm_cell.png" alt="LSTM cell diagram">
  <figcaption>Internal structure of an LSTM cell.</figcaption>
</figure>

### Transformers: Attention is All You Need

RNNs and LSTMs process data sequentially, which limits parallelization. **Transformers** (2017) changed the game by using the **Attention** mechanism. Instead of compressing the past into a vector, attention allows the model to "look" directly at any point in the past to understand the present.

$$\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

This enables massive parallelization and better capture of very long-term dependencies.

<figure>
  <img src="/assets/images/timeseries_attention_mechanism.png" alt="Attention mechanism diagram">
  <figcaption>The Scaled Dot-Product Attention mechanism.</figcaption>
</figure>

---

## Part 2: Practice - From Theory to Code

We will use **PyTorch** to compare these models. The complete code is available on the [GitHub repository](https://github.com/julienguegan/notebooks_blog).

### Level 1: The Basics (Sine Wave)

Let's start with the "Hello World" of time series: predicting a sine function. It's simple, clean and noise-free.

```python
# Generate a simple sine wave
def generate_sine_wave(freq, sample_rate, duration):
    t = np.linspace(0, duration, int(sample_rate * duration))
    y = np.sin(2 * np.pi * freq * t)
    return t, y
```

LSTM and Transformer models learn this task without difficulty.

<figure>
  <img src="/assets/images/timeseries_sine_wave_prediction.png" alt="Sine Wave Prediction">
  <figcaption>On a simple series, both models are perfect.</figcaption>
</figure>

### Level 2: Increasing Complexity

Reality is rarely as clean as a sine wave. Let's add some spice with harmonics, trends, modulation and even chaos.

#### Case A: Multi-frequencies and Trend

Here, we combine several waves, add a linear and quadratic trend, as well as noise.

```python
def generate_complex_wave_v1(sample_rate, duration):
    # ... (combination of sines, harmonics and trends)
    y += 0.02 * t + 0.001 * t**2 # Trend
    y += noise_level * np.random.randn(len(t)) # Noise
    return t, y
```

<figure>
  <img src="/assets/images/timeseries_complex_wave_1.png" alt="Complex Wave 1 Prediction">
  <figcaption>The LSTM follows the trend well, the Transformer better captures rapid peaks.</figcaption>
</figure>

#### Case B: Modulation (AM/FM) and Bursts

A radio or audio signal often looks like this: amplitude and frequency modulation, with sudden "bursts".

<figure>
  <img src="/assets/images/timeseries_complex_wave_2.png" alt="Modulated Wave Prediction">
  <figcaption>Modulation tests the model's ability to adapt to regime changes.</figcaption>
</figure>

#### Case C: Chaos and Seasonality

Let's mix seasonal cycles (like annual sales) with a chaotic component (Lorenz attractor type). This is the nightmare of classical linear models.

<figure>
  <img src="/assets/images/timeseries_complex_wave_3.png" alt="Chaos Prediction">
  <figcaption>Even with deterministic chaos, Deep Learning models manage to anticipate the dynamics.</figcaption>
</figure>

---

## Part 3: "Real" Case - Coffee vs Productivity ☕️

To finish, let's take a more... pragmatic example. Imagine we want to predict a **developer's productivity** (on an arbitrary scale) based on several factors:

1.  **Circadian Rhythm**: We sleep at night (low productivity).
2.  **Caffeine**: Productivity peaks after coffee at 8am and 2pm (with exponential decline).
3.  **Weekend**: Lower (or different) productivity.
4.  **Production Bugs**: Brutal and random drops in productivity ("Server Outages").

```python
def generate_coffee_productivity(days=100):
    # ...
    # Caffeine peaks
    coffee_effect[idx_8am:idx_8am+5] += np.exp(-np.arange(5)/2) * 2.0

    # Random drops (Bugs)
    outages[idx:idx+4] = -2.0

    productivity = 5 + 2 * circadian + coffee_effect + outages
    return t, productivity
```

We train our models on 30 days of this simulated developer's life. Here's the result:

<figure>
  <img src="/assets/images/timeseries_coffee_and_productivity.png" alt="Coffee Productivity Prediction">
  <figcaption>The model learns sleep cycles and coffee boosts, but obviously cannot predict random bugs (the brutal drops) that have no precursors in the past!</figcaption>
</figure>

This is a crucial point: **a model cannot predict pure randomness**. It learns the circadian rhythm and the coffee effect (which are regular), but logically fails to anticipate random server outages. However, it readjusts very quickly after the incident.

## Conclusion

Modern time series require powerful tools. While statistical methods (ARIMA) remain valid for simple cases, Deep Learning (LSTM, Transformers) excels as soon as complexity, non-linearity and dimensionality increase.

To go further, we could explore:

- **Specialized temporal Transformers** (Informer, Autoformer).
- Adding **exogenous variables** (e.g., giving the model the time of day or the amount of coffee consumed as explicit input).

---

[![Generic badge](https://img.shields.io/badge/written_with-Jupyter_notebook-orange.svg?style=plastic&logo=Jupyter)](https://jupyter.org/try) [![Generic badge](https://img.shields.io/badge/License-MIT-blue.svg?style=plastic)](https://lbesson.mit-license.org/) [![Generic badge](https://img.shields.io/badge/code_access-github-black.svg?style=plastic&logo=github)](https://github.com/julienguegan/notebooks_blog/blob/main/script/time_series.py)
