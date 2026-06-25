# Tweet Sentiment Analysis: Text Representation Strategies Compared

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A complete NLP project that implements and compares multiple text representation strategies for **three-class sentiment classification of tweets** (Negative / Neutral / Positive), combining contextual embeddings, unsupervised clustering, and neural network classification.

## 📋 Abstract

This project explores the question at the heart of every NLP classification task: *how do you turn variable-length text into fixed-size numerical features that a classifier can use?*

Starting from a balanced dataset of **17,910 English tweets**, we evaluate four fundamentally different approaches — from static Word2Vec embeddings to a zero-shot pre-trained transformer — and measure how each representation strategy affects the final accuracy. The main contribution is a novel pipeline that clusters BERTweet contextual embeddings via K-Medoids before feeding them to an MLP, and an accompanying sensitivity analysis over the clustering parameter K.

## 🎯 Objectives

- Extract and compare different types of text representations for tweet sentiment classification
- Implement a custom pipeline: contextual embeddings → K-Medoids clustering → MLP
- Establish a Word2Vec baseline with static, non-contextual embeddings
- Benchmark zero-shot performance of a pre-trained sentiment transformer
- Analyse how the K-Medoids parameter K affects classification accuracy

## 📊 Dataset

The dataset contains **17,910 English tweets** pre-labelled with sentiment:

| Split | Samples | Samples per class |
|---|---|---|
| Training set | 14,328 | ~4,776 |
| Test set | 3,582 | 1,194 |

- **Classes**: Negative (−1) · Neutral (0) · Positive (+1)
- **Balance**: perfectly balanced — equal support per class in both splits
- **Format**: CSV with columns `clean_text` (pre-cleaned tweet) and `category` (label)

## 🔧 Methodology

### Part 1 – Main Method (Sciacca): BERTweet + K-Medoids + MLP

The reference pipeline transforms each tweet into a fixed-size vector through three stages:

**1. Contextual Embedding Extraction**

The pre-trained model `vinai/bertweet-base` — a RoBERTa variant optimised for social media — produces a 768-dimensional vector for each token. Unlike static embeddings, these vectors depend on the full sentence context: the word "great" has a different representation in "great movie" and "not great at all".

**2. K-Medoids Clustering**

The token vectors of each tweet are grouped into K clusters by semantic proximity. For each cluster, the mean of its member vectors is computed. The representative vectors are then concatenated in the order of their first appearance in the original text, partially preserving grammatical structure. The result is a fixed-size vector of K × 768 dimensions.

```
Tweet (text)
     │
     ▼
BERTweet tokenizer
     │
     ▼
Contextual embedding matrix [n_tokens × 768]   ← variable size
     │
     ▼
K-Medoids (K=5) — groups semantically similar tokens
     │
     ├── Cluster 0 → mean → vector 768
     ├── Cluster 1 → mean → vector 768
     ├── Cluster 2 → mean → vector 768
     ├── Cluster 3 → mean → vector 768
     └── Cluster 4 → mean → vector 768
     │
     ▼
Concatenation in chronological order
[3840]  ← always fixed size
     │
     ▼
MLPClassifier (256 neurons)
     │
     ▼
Sentiment: -1 / 0 / +1
```

**3. MLP Classification**

A single-hidden-layer MLP (256 neurons, ReLU, Adam optimiser) is trained on the K-Medoids representations. Early stopping (patience = 5 epochs) prevents overfitting.

---

### Part 2 – Comparison Framework (Comis)

Three additional experiments are run against the same train/test split:

**Method 1 – Word2Vec + MLP (Baseline)**
Static `word2vec-google-news-300` embeddings (300 dimensions). Each tweet is represented by the mean of its word vectors. Same MLP architecture as the main method. Serves as the lower bound.

**Method 2 – Pre-trained BERTweet (Zero-Shot)**
Direct application of `finiteautomata/bertweet-base-sentiment-analysis` — built on the same `vinai/bertweet-base` backbone but already fine-tuned for sentiment on SemEval 2017 tweets. No training on our dataset. Labels `POS`/`NEU`/`NEG` are remapped to `+1`/`0`/`-1`.

**Method 3 – K Sensitivity Analysis**
Sciacca's full pipeline is re-run for K ∈ {1, 3, 5, 7, 10}. A separate MLP is trained for each K to find the optimal clustering parameter.

## 📈 Results

### Main Comparative Table

| Method | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Pre-trained BERTweet (zero-shot) | **0.7465** | **0.7460** | **0.7465** | **0.7398** |
| BERTweet K-Medoids (K=3) + MLP | 0.7225 | – | – | – |
| BERTweet K-Medoids (K=1) + MLP | 0.7211 | – | – | – |
| BERTweet K-Medoids (K=5) + MLP | 0.7058 | 0.7151 | 0.7058 | 0.7081 |
| BERTweet K-Medoids (K=10) + MLP | 0.7058 | – | – | – |
| BERTweet K-Medoids (K=7) + MLP | 0.6999 | – | – | – |
| Word2Vec + MLP | 0.6547 | 0.6634 | 0.6547 | 0.6571 |

*Test set: 3,582 tweets, 1,194 per class.*

### K Sensitivity Analysis

| K | Vector dimension | Accuracy |
|---|---|---|
| 1 | 768 | 0.7211 |
| **3** | **2,304** | **0.7225** ← optimal |
| 5 | 3,840 | 0.7058 |
| 7 | 5,376 | 0.6999 |
| 10 | 7,680 | 0.7058 |

## 🔍 Key Findings

1. **Contextual embeddings are decisive.** All BERTweet-based methods outperform Word2Vec — even K=1 (simple mean of contextual embeddings) improves accuracy by nearly 7 percentage points. The difference lies in embedding quality, not in the aggregation method.

2. **A domain-specialised model wins without training.** The zero-shot pre-trained transformer achieves the highest accuracy (74.65%), confirming that domain-specific fine-tuning is the most powerful factor.

3. **K=5 is not the optimal choice.** The sensitivity analysis shows that lower values (K=3 at 72.25%, K=1 at 72.11%) outperform K=5 (70.58%). Adding more clusters beyond K=3 introduces redundancy and hurts the MLP.

4. **The Neutral class is the hardest for all methods.** Neutral tweets lack the strong lexical markers of positive and negative text, causing systematic misclassification across every approach tested.

## 📦 Requirements

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- scikit-learn
- scikit-learn-extra
- gensim
- numpy 1.26.4
- pandas
- matplotlib
- seaborn
- emoji 0.6.0

### Installation

```bash
pip install torch transformers scikit-learn gensim pandas matplotlib seaborn
pip install numpy==1.26.4 scikit-learn-extra emoji==0.6.0
```

> **Note:** `numpy==1.26.4` is required because `scikit-learn-extra` is compiled with Cython against numpy 1.x. `emoji==0.6.0` is an internal dependency of the BERTweet tokenizer.

## 🚀 Usage

The project runs as a Jupyter / Google Colab notebook:

1. Place `train_set.csv` and `test_set.csv` in the same directory as the notebook (or in `/content/` on Colab).
2. Open `Progetto_NLP.ipynb` and run all cells sequentially.
3. The notebook is self-contained: it installs dependencies, downloads models, trains classifiers, and produces all results and visualisations.

> **GPU recommended.** BERTweet embedding extraction is substantially faster on CUDA. On Google Colab, select *Runtime → Change runtime type → T4 GPU*.

## 🏗️ Project Structure

```
├── Progetto_NLP.ipynb    # Main notebook (full pipeline + comparison framework)
├── train_set.csv         # Training set (14,328 tweets)
├── test_set.csv          # Test set (3,582 tweets)
└── README.md             # This file
```

## 💡 Future Improvements

- **Fine-tuning BERTweet** on the training set: updating the transformer weights on our specific labels would likely push accuracy beyond the 74.65% zero-shot ceiling.
- **Alternative clustering algorithms**: compare K-Medoids with K-Means or hierarchical clustering to evaluate the impact of centroid selection.
- **Larger datasets**: re-evaluating K on a larger dataset may reveal whether K>3 becomes beneficial when more training samples are available.
- **Ensemble methods**: combining predictions from multiple K values or multiple architectures for improved robustness.

## 👥 Authors

- **Lorenzo Comis** — [GitHub](https://github.com/LorenzoComis-git)
- **Alessandro Sciacca**

**Academic Year**: 2025-2026  
**Course**: Natural Language Processing  
**Institution**: Department of Mathematics and Computer Science, University of Catania

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*For questions or suggestions, feel free to open an issue on GitHub.*
