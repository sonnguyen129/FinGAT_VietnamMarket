# FinGAT: Financial Graph Attention Networks for Recommending Top-K Profitable Stocks

**Authors:** Yi-Ling Hsu\*, Yu-Che Tsai\*, Cheng-Te Li, *Member, IEEE*
*\*Equal contribution*

**Published:** IEEE Transactions on Knowledge and Data Engineering (TKDE), 2021
**arXiv:** 2106.10159v1 [cs.LG] 18 Jun 2021
**Code:** https://github.com/Roytsai27/Financial-GraphAttention

**Affiliations:**
- Yi-Ling Hsu — Master Program in Statistics, National Taiwan University, Taipei, Taiwan
- Yu-Che Tsai — Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan
- Cheng-Te Li — Institute of Data Science, National Cheng Kung University, Tainan, Taiwan

---

## Abstract

Financial technology (FinTech) has drawn much attention among investors and companies. While conventional stock analysis in FinTech targets at predicting stock prices, less effort is made for profitable stock recommendation. Besides, in existing approaches on modeling time series of stock prices, the relationships among stocks and sectors (i.e., categories of stocks) are either neglected or pre-defined. Ignoring stock relationships will miss the information shared between stocks while using pre-defined relationships cannot depict the latent interactions or influence of stock prices between stocks.

In this work, we aim at recommending the top-K profitable stocks in terms of return ratio using time series of stock prices and sector information. We propose a novel deep learning-based model, **Financial Graph Attention Networks (FinGAT)**, to tackle the task under the setting that no pre-defined relationships between stocks are given. The idea of FinGAT is three-fold:

1. We devise a hierarchical learning component to learn short-term and long-term sequential patterns from stock time series.
2. A fully-connected graph between stocks and a fully-connected graph between sectors are constructed, along with graph attention networks, to learn the latent interactions among stocks and sectors.
3. A multi-task objective is devised to jointly recommend the profitable stocks and predict the stock movement.

Experiments conducted on Taiwan Stock, S&P 500, and NASDAQ datasets exhibit remarkable recommendation performance of our FinGAT, comparing to state-of-the-art methods.

**Index Terms:** profitable stock recommendation, graph attention networks, stock movement prediction, sector information

---

## 1. Introduction

The stock market has grown swiftly in these years, and trading stocks have become one of the most attractive financial instruments for investors. Investing in the stock market is highly profitable and easy to get started. However, investing stocks usually involves extremely high risk, which makes drawing up a proper investment plan a crucial task. Previously, people tend to empirically choose stocks by their financial knowledge or expertise. As financial technology (FinTech) is now in widespread use, people come up with statistical inference models to forecast the dynamic movement of stock prices [23]. Techniques of machine learning and deep learning are investigated and applied in industries, which has shown remarkable success in different stock markets, such as S&P 500 [15] and NASDAQ [12].

In predicting stock prices, typical methods such as Auto Regression-based methods [2], [11] treat time series indicators (e.g., stock price) as a linear stochastic process. However, a stock time series usually appears in a dynamic nonlinear process. Regarding this drawback, deep learning methods such as Recurrent Neural Network (RNN) [1], [5] project time series into a high-dimensional space to obtain its sequential-level representation. Attention mechanism [19] has been applied to consider the varying importance of each timestamp by giving learnable attentive weights. With the rise of graph neural network, recent studies [6], [15] incorporate the relationships (e.g., upstream, downstream, and sector) between stocks to form graphs, which are further used to pass financial knowledge between stocks so as to distill graph-level features. A key idea is that the stock price is not only affected by the company itself, but also determined by the global trend of the market or the financial situation of its competitors. Hence, modeling with graph neural networks seamlessly unifies the information of the targeted company as well as the correlated companies based on the constructed graphs.

In contrast with stock price prediction, since the investors care more about identifying stocks that can bring higher return in the future, few studies have attempted to recommend profitable stocks. Convolutional and recurrent neural networks are effective in extracting long-term and short-term sequential features from the time series of stock prices and producing promising recommendation performance [13], [27]. Nevertheless, it is still worthwhile to study how to model the relationships between stocks for profitable stock recommendation.

It is challenging to model the relationships between stocks (i.e., listed companies) in recommending profitable stocks:

1. **Confidentiality.** The data on company-company relationships, such as "investing", "member of", "subsidiary", and "complies with", is difficult to access due to confidentiality agreement, security issues, or privacy concerns. Manually collecting the relationship data could be either incomplete or labeling-bias.
2. **Dynamic.** The relationships between listed companies are dynamic. Two companies can change their relationships between competition and support with time. It is less possible to track the evolution of all their relationships.
3. **Inter-sector latent relations.** The latent relationships between sectors (i.e., stock categories), such as the underlying correlation among "oil", "textile", and "gold" sectors, can also affect the rise and fall of stock prices. Such inter-sector relations are usually implicit and hard to be concretely defined, comparing to the explicit company-company relations, which are termed *intra-sector relation*.

> **Figure 1:** A toy example of intra-sector relations and inter-sector relations.
> Stocks within the same sector (Oil, Textile, Gold) are connected by *intra-sector* (solid) edges. Sectors are connected by *inter-sector* (dashed) edges.

In the real world, stocks within a sector (e.g., oil) usually have a similar price movement trend. Prices of related sectors (e.g., gold and textile) can be influenced by stock prices in the oil sector. For example, the growth of oil prices usually involves the happening of inflation phenomena and leads to the increasing uncertainty of economic development. Since gold is a financial item against inflation, the demand for gold tends to increase, which is followed by the growth of oil prices [31].

In this paper, we aim at recommending the most profitable stocks. Given the historical time-series data of stock prices for a set of listed companies, our goal is to recommend stocks that can bring the highest return by investing them in the next day. To better represent each stock, we will model not only the sequential patterns hidden in time series, but also the hierarchical influence between stocks at both stock and sector levels.

We propose **FinGAT**, which consists of three main phases:

1. **Stock-level feature learning.** Extract a variety of features per stock per day, and exploit attentive GRU to learn short-term (single-week) sequential features. Construct a fully-connected graph of same-sector stocks and apply GAT to learn latent intra-sector relations.
2. **Sector-level feature learning.** Create a weekly aggregation layer that combines short-term embeddings via attentive GRU. A graph pooling mechanism generates the sector embedding from same-sector stocks. GAT is again applied to learn latent inter-sector relations.
3. **Multi-task learning.** Profit is influenced by two highly-correlated factors: stock price return (real value) and stock movement (binary value). A multi-task objective jointly optimizes both.

**Contributions:**

- **Conceptual.** Recommend the most profitable stocks by modeling hierarchical correlation: intra-sector, inter-sector, and stock-sector relations. Novel: learn latent relations rather than rely on pre-defined knowledge.
- **Technical.** A novel multi-task graph neural network-based model, FinGAT, that learns sequential patterns in financial time series and hierarchical influence among stocks and sectors.
- **Empirical.** Experiments on Taiwan Stock, S&P 500, and NASDAQ show FinGAT outperforms state-of-the-art by 17% and 13%. Also performs well even without sector information. Provides interpretability via attention weights.

**Paper organization:** Section 2 reviews relevant studies. Section 3 presents the problem statement. Section 4 describes FinGAT. Section 5 reports experimental results. Section 6 concludes.

---

## 2. Related Work

Typical methods for stock prediction include **ARIMA** [2] and **SVR** [4]. ARIMA considers the linear combination of historical stock prices while SVR treats the stock price at each timestamp independently. By learning sequential patterns, **RNN-based models** [1], [5], including LSTM and GRU, improve the prediction performance. **SFM** [29] further improves LSTM with memory network and modeling multi-frequency trading patterns. **Attention-based models** [19] further learn how the next prediction is attended by historical hidden states of stock time series, and produce better results.

Despite that RNN-based models achieve great success in sequential modeling, the performance drops significantly as the length of the sequence increases [14]. To capture the long-term dependency of stock movement, **FineNet** [27] utilizes two dilated convolution neural networks to jointly model both long-term and short-term sequential patterns.

While most existing methods aim at minimizing pointwise loss, a recent RNN-based model, **Rank-LSTM** [13], utilizes pairwise ranking-aware loss, along with pointwise regression loss, to recommend profitable stocks. We will compare the proposed FinGAT with Rank-LSTM as it is a state-of-the-art. Ding et al. [10] propose the *extreme value loss (EVL)* that enables model to detect extreme stock events.

Some recent advances attempt to model how stocks are affected by one another through learning features from the relationships between stocks. Given a graph constructed from a set of pre-defined relationships of investment facts between listed companies, **temporal graph convolutional networks** [6], [13], [21] are utilized to extract graph-based stock interaction features. **HATS** [15] also relies on the pre-defined graph depicting the relations between stocks, but it further learns the embeddings of different types of relationships by considering their hierarchical structure.

However, we argue that it is unrealistic to presume that the relationships between stocks are always accessible. The relationships are usually hidden due to business concerns. In addition, some influence of price movement between stocks cannot be reflected by their relationships. Our work aims at learning the **latent intra-sector and inter-sector relations** between stocks. The proposed FinGAT does not rely on any pre-defined relationships.

---

## 3. Problem Statement

We are targeting that investors have enough funding but lack insights in deciding which stocks are more worthy to invest among all listed companies.

We define the **return ratio** of stock $s_q$ at the $j$-th day of week $i$, denoted by $R^{s_q}_{ij}$, by considering how much an investor can earn by investing one dollar from the previous day $j-1$:

$$R^{s_q}_{ij} = \frac{p^{s_q}_{ij} - p^{s_q}_{i(j-1)}}{p^{s_q}_{i(j-1)}} \tag{1}$$

where $p^{s_q}_{ij}$ is the stock price for stock $s_q$ at the $j$-th day in week $i$.

Let $S = \{s_1, s_2, ..., s_n\}$ denote the universe set of $n$ stocks, and let $\mathbf{v}^{s_q}_{ij}$ denote the feature vector of stock $s_q$ at the $j$-th day of week $i$. Each stock $s_q$ has its corresponding sector, denoted as $\pi_c$, which is an area of the economy that businesses share the same or related products/services. A sector can contain multiple stocks.

Given the feature matrix
$$D^{s_q}_i = \{\mathbf{v}^{s_q}_{i(j-d)}, \mathbf{v}^{s_q}_{i(j-d+1)}, ..., \mathbf{v}^{s_q}_{i(j-1)}\}$$
derived from the $(j-d)$-th to $(j-1)$-th day in week $i$ (where $d$ is the number of past days), our goal is to:

1. Predict the return ratio $R^{s_q}_{(i+1)1}$ of every stock $s_q \in S$.
2. Recommend a list of **top-K** stocks for the first day of week $i+1$.

---

## 4. The Proposed FinGAT Model

> **Figure 2:** Architecture of the proposed FinGAT model.
> Three main components: (1) Stock-Level Modeling (Short-term Sequential Learning + Intra-Sector Relation Modeling + Long-term Sequential Learning); (2) Sector-Level Modeling (Intra-Sector Graph Pooling + Inter-Sector Relation Modeling); (3) Model Learning (Fusion + Task-Specific Layers for Profitable Stock Ranking & Stock Movement prediction).

FinGAT consists of three main components: (1) **stock-level modeling**, (2) **sector-level modeling**, and (3) **model training**.

### 4.1 Stock-level Modeling

#### Feature Extraction

We define the initial features of each stock at one day. Basic daily features:
- Opening price $\text{open}_j$ and closing price $\text{close}_j$
- Highest price $\text{high}_j$ and lowest price $\text{low}_j$
- Adjusted closing price $\text{adjclose}_j$
- Return ratio

We also define hand-crafted features:

**Price-ratio features:**
$$F_\mu = \frac{\mu_j}{\text{close}_j} - 1 \tag{2}$$
where $\mu \in \{\text{open}, \text{high}, \text{low}\}$.

**Moving-average features:**
$$F_\phi = \frac{\frac{1}{\phi} \sum_{k=0}^{\phi-1} \text{adjclose}_{j-k}}{\text{adjclose}_{j-1}} \tag{3}$$
where $\phi \in \{5, 10, 15, 20, 25, 30\}$. (Numerator is the $\phi$-day moving average of adjusted close.)

We concatenate these feature values as the initial feature vector $\mathbf{v}^{s_q}_{ij}$ for stock $s_q$ at the $j$-th day in week $i$.

#### Short-term Sequential Learning

We exploit GRU to learn short-term sequential features. The sequence of feature vectors $\mathbf{v}^{s_q}_{ij}$ within a week is the input. GRU's last hidden-state vector $\mathbf{h}^{s_q}_{ij}$ is the output for week $i$:

$$\mathbf{h}^{s_q}_{ij} = \text{GRU}(\mathbf{v}^{s_q}_{ij}, \mathbf{h}^{s_q}_{i(j-1)}) \tag{4}$$

A feed-forward neural network-based **attention mechanism** [20] learns dynamic weights and aggregates day-wise hidden states to encode week $i$'s sequential patterns. Let $H_i = \{\mathbf{h}^{s_q}_{i1}, \mathbf{h}^{s_q}_{i2}, ..., \mathbf{h}^{s_q}_{id}\}$ be the input. The attentive representation:

$$\mathbf{a}^{s_q}_i = \text{Attention}(H_i) = \sum_j \alpha^{s_q}_j \mathbf{h}^{s_q}_{ij} \tag{5}$$

$$\alpha^{s_q}_j = \sigma\!\left(\tanh\!\left(W_0 \mathbf{h}^{s_q}_{ij}\right)\right) \tag{6}$$

where $W_0$ is learnable, and $\sigma$ is the **softmax** function (over $j$). Note $j$ refers to each day within week $i$.

#### Intra-sector Relation Modeling

We aim to model latent relationships between same-sector stocks. For each sector $\pi_c$, we create a **fully-connected graph** $G_{\pi_c} = (M_{\pi_c}, E_{\pi_c})$ where $M_{\pi_c}$ is the set of listed companies in $\pi_c$. For each $s_q \in M_{\pi_c}$, we create an edge to every other $s_r \in M_{\pi_c}$ where $s_r \neq s_q$. The initial node feature is $\mathbf{a}^{s_q}_i$.

**Graph Attention Network (GAT)** [28] is adopted because relationship strengths vary:

$$\text{GAT}(G_{\pi_c}; s_q) = \text{ReLU}\left(\sum_{s_n \in \Gamma(s_q)} \beta_{qn} W_1 \mathbf{a}^{s_n}_i\right) \tag{7}$$

where $\beta_{qn}$ is the attention weight from stock $s_n$ to $s_q$, $\Gamma(s_q)$ returns the neighbors, and $W_1$ is learnable.

Attention weights:
$$\beta_{qn} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{r}^\top [W_2 \mathbf{a}^{s_q}_i \| W_2 \mathbf{a}^{s_n}_i]\right)\right)}{\sum_{s_n \in \Gamma(s_q)} \exp\left(\text{LeakyReLU}\left(\mathbf{r}^\top [W_2 \mathbf{a}^{s_q}_i \| W_2 \mathbf{a}^{s_n}_i]\right)\right)} \tag{8}$$

where $\mathbf{r}$ is a learnable projection vector, $\|$ denotes concatenation, and $W_2$ is learnable.

The graph-based representation $\mathbf{g}^{s_q}_i = \text{GAT}(G_{\pi_c}; s_q)$ encodes intra-sector relations.

#### Long-term Sequential Learning

Since stock price is affected by both short-term and long-term movements [27], we aggregate weekly embeddings. Two kinds of temporal information:
- $\mathbf{a}^{s_q}_i$: encodes primitive long-term sequential features
- $\mathbf{g}^{s_q}_i$: incorporates intra-sector relations

For past $t$ weeks:

$$U^G_i(s_q) = \{\mathbf{g}^{s_q}_{i-t}, \mathbf{g}^{s_q}_{i-t+1}, ..., \mathbf{g}^{s_q}_{i-1}\}$$
$$U^A_i(s_q) = \{\mathbf{a}^{s_q}_{i-t}, \mathbf{a}^{s_q}_{i-t+1}, ..., \mathbf{a}^{s_q}_{i-1}\} \tag{9}$$

Apply attentive GRU separately:

$$\tau^G_i(s_q) = \text{Attention}(U^G_i(s_q))$$
$$\tau^A_i(s_q) = \text{Attention}(U^A_i(s_q)) \tag{10}$$

### 4.2 Sector-level Modeling

#### Intra-sector Graph Pooling

Generate a sector embedding from same-sector stocks using **element-wise max-pooling**:

$$\mathbf{z}_{\pi_c} = \text{MaxPool}\left(\{\tau^G_i(s_q) \mid \forall s_q \in M_{\pi_c}\}\right) \tag{11}$$

where $\text{MaxPool}(X) = [\max(\{x_1 | \forall \mathbf{x} \in X\}), ..., \max(\{x_\epsilon | \forall \mathbf{x} \in X\})]$.

We use max-pooling for simplicity (no learnable parameters). A set of sector embeddings $Z_\pi = \{\mathbf{z}_{\pi_1}, \mathbf{z}_{\pi_2}, ..., \mathbf{z}_{\pi_c}\}$ is obtained.

#### Inter-sector Relation Modeling

Construct a **fully-connected graph** $G_\pi = (Z_\pi, E_\pi)$ where every sector pair is directly connected. Apply GAT:

$$\tau_i(\pi_c) = \text{GAT}(G_\pi, \pi_c) \tag{12}$$

The derived $\tau_i(\pi_c)$ encodes how sectors influence each other.

### 4.3 Model Learning

#### Embedding Fusion

Combine $\tau^A_i(s_q)$, $\tau^G_i(s_q)$, $\tau_i(\pi_c)$ via fusion:

$$\tau^F_i(s_q) = \text{ReLU}\left([\tau^G_i(s_q) \| \tau^A_i(s_q) \| \tau_i(\pi_c)] W_f\right) \tag{13}$$

where $s_q$ belongs to $\pi_c$, $W_f$ is learnable.

#### Multi-Task Learning

Recommending the most profitable stocks splits into two correlated parts:
1. **Ranking** stocks based on predicted return ratios
2. **Movement prediction** (binary up/down)

Rather than pointwise loss (e.g., MSE) that cannot reflect profitability, we jointly optimize:
- **Pairwise ranking-aware loss** [13] for ranking
- **Cross-entropy loss** for movement

Predictions:
$$\hat{y}^{\text{return}}_i(s_q) = \mathbf{e}_1^\top \tau^F_i(s_q) + b_1$$
$$\hat{y}^{\text{move}}_i(s_q) = \varphi\left(\mathbf{e}_2^\top \tau^F_i(s_q) + b_2\right) \tag{14}$$

where $\varphi$ is sigmoid.

To generate daily predictions, we use a daily sliding window (details in Section 5.1).

**Total loss:**
$$\mathcal{L}_{\text{FinGAT}} = (1 - \delta)\mathcal{L}_{\text{rank}} + \delta \mathcal{L}_{\text{move}} + \lambda \|\Theta\|^2 \tag{15}$$

where $\Theta$ depicts all learnable weights, $\lambda$ is the L2 regularization hyperparameter.

**Component losses:**
$$\mathcal{L}_{\text{rank}} = \sum_i \sum_{s_q} \sum_{s_k} \max\left(0, -\hat{\Delta} \times \Delta\right)$$
$$\text{where } \hat{\Delta} = \hat{y}^{\text{return}}_i(s_q) - \hat{y}^{\text{return}}_i(s_k), \quad \Delta = y^{\text{return}}_i(s_q) - y^{\text{return}}_i(s_k)$$

$$\mathcal{L}_{\text{move}} = -\sum_i \sum_{s_q} y^{\text{move}}_i \log(\hat{y}^{\text{move}}_i(s_q)) + (1 - y^{\text{move}}_i) \log(1 - \hat{y}^{\text{move}}_i(s_q)) \tag{16}$$

$\delta$ balances the two tasks.

---

## 5. Evaluation

We answer five evaluation questions:

- **EQ1:** Can FinGAT outperform state-of-the-art models on top-K stock recommendation?
- **EQ2:** Will FinGAT still perform well if no sector information is given?
- **EQ3:** Does each component of FinGAT effectively contribute?
- **EQ4:** How do hyperparameters affect performance?
- **EQ5:** What can intra-sector and inter-sector graph attention weights capture?

### 5.1 Evaluation Settings

**Datasets:** Three real-world datasets:
- **Taiwan Stock** (https://www.twse.com.tw/en/page/listed/listed_company/new_listing.html)
- **S&P 500** (https://datahub.io/core/s-and-p-500)
- **NASDAQ** (https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)

For Taiwan Stock and S&P 500: 60% train (579 days), 20% validation (193 days), 20% testing (193 days). Every consecutive **16 trading days** is one data instance: 3 weeks (5 days/week) as training input, the 16th day as prediction target. Sliding window with 15 days. For NASDAQ, follow RankLSTM [13]'s setting.

**Table 1: Statistics of stock datasets.**

| Market | # Stocks | # Sectors | # Training Days | # Validation Days | # Testing Days |
|--------|---------:|----------:|----------------:|------------------:|---------------:|
| Taiwan Stock | 100 | 5 | 579 | 193 | 193 |
| S&P 500 | 424 | 9 | 579 | 193 | 193 |
| NASDAQ | 1026 | 112 | 756 | 252 | 237 |

**Evaluation Metrics:**

Let $R^S_{ij} = \{R^{s_1}_{ij}, ..., R^{s_n}_{ij}\}$ be ground-truth return ratios; $\hat{R}^S_{ij}$ predicted. Stocks with higher return ratio are ranked at top. $\mathcal{L}@K(R^S_{ij})$ and $\mathcal{L}@K(\hat{R}^S_{ij})$ are predicted/ground-truth top-K lists.

Since our goal is recommendation, error metrics (MAE, RMSE) are not adopted. We use:

- **Mean Reciprocal Rank (MRR@K):**
$$\text{MRR@K} = \frac{1}{K} \sum_{s_q \in \mathcal{L}@K(\hat{R}^S_{ij})} \frac{1}{\text{rank}(R^{s_q}_{ij})} \tag{17}$$

- **Precision@K:**
$$\text{Precision@K} = \frac{|\mathcal{L}@K(\hat{R}^S_{ij}) \cap \mathcal{L}@K(R^S_{ij})|}{K} \tag{18}$$

- **Accuracy (ACC):** Number of correct binary movement predictions / total predictions.

Higher = better. Average over **10 runs** is reported. Same evaluation procedure applied to FinGAT and competing methods.

**Competing Methods:**

1. **MLP** [26]: 2 hidden layers (32, 8 dims).
2. **GRU** [7]: 1 layer, 32-dim.
3. **GRU+Att** [9]: 32-dim GRU + attention layer.
4. **FineNet** [27]: SOTA joint CNN+RNN (32-dim + 16-dim conv layers) for short/long-term patterns.
5. **RankLSTM** [13]: SOTA temporal graph conv (16-dim) + pairwise ranking loss. Search $\alpha \in \{0.01, 0.1, 1, 10\}$.

**FinGAT Settings:**
- GRU and GAT hidden dim: 16
- Learning rate: search $\{0.0005, 0.001, 0.005\}$
- Batch size: 128
- $\delta = 0.01$
- $\lambda = 0.0001$
- Optimizer: Adam [16]
- Framework: PyTorch + PyTorch Geometric
- Hardware: Nvidia GeForce GTX 1080 Ti

### 5.2 Experimental Results

#### Main Results

**Table 2: Main experimental results by varying top-K, K ∈ {5, 10, 20}.**

##### Taiwan Stock

| Model | MRR@5 | Prec@5 | MRR@10 | Prec@10 | MRR@20 | Prec@20 | ACC |
|-------|------:|-------:|-------:|--------:|-------:|--------:|----:|
| MLP | 0.2842 | 0.0500 | 0.5753 | 0.1022 | 1.0114 | 0.1992 | 0.4514 |
| GRU | 0.3115 | 0.0622 | 0.6222 | 0.1272 | 1.0639 | 0.2053 | 0.4812 |
| GRU+Att | 0.3435 | 0.0811 | 0.6736 | 0.1417 | 1.1779 | 0.2131 | 0.4948 |
| FineNet | 0.3742 | 0.0867 | 0.7002 | 0.1572 | 1.2500 | 0.2206 | 0.5295 |
| RankLSTM | 0.3962 | 0.1011 | 0.7838 | 0.1717 | 1.3298 | 0.2456 | 0.5539 |
| **FinGAT** | **0.4391** | **0.1133** | **0.8479** | **0.2022** | **1.4106** | **0.2756** | **0.5682** |
| Improv. | 10.83% | 12.08% | 8.18% | 17.76% | 6.08% | 12.21% | 2.58% |

##### S&P 500

| Model | MRR@5 | Prec@5 | MRR@10 | Prec@10 | MRR@20 | Prec@20 | ACC |
|-------|------:|-------:|-------:|--------:|-------:|--------:|----:|
| MLP | 0.0844 | 0.0172 | 0.1828 | 0.0215 | 0.3886 | 0.0594 | 0.535 |
| GRU | 0.1158 | 0.0266 | 0.2229 | 0.0366 | 0.4390 | 0.0685 | 0.5342 |
| GRU+Att | 0.1321 | 0.0301 | 0.2513 | 0.0516 | 0.4813 | 0.0849 | 0.5353 |
| FineNet | 0.1502 | 0.0387 | 0.2965 | 0.0602 | 0.5392 | 0.0909 | 0.539 |
| RankLSTM | 0.1736 | 0.0398 | 0.3034 | 0.0597 | 0.5411 | 0.0911 | 0.5411 |
| **FinGAT** | **0.1974** | **0.0419** | **0.3357** | **0.0677** | **0.5687** | **0.0989** | **0.5425** |
| Improv. | 13.71% | 5.28% | 10.65% | 12.46% | 5.10% | 8.56% | 0.26% |

##### NASDAQ

| Model | MRR@5 | Prec@5 | MRR@10 | Prec@10 | MRR@20 | Prec@20 | ACC |
|-------|------:|-------:|-------:|--------:|-------:|--------:|----:|
| MLP | 6.13e-3 | 1.12e-3 | 8.95e-3 | 2.01e-3 | 2.10e-2 | 3.56e-3 | 0.1374 |
| GRU | 9.31e-3 | 3.85e-3 | 2.14e-2 | 4.89e-3 | 3.76e-2 | 6.03e-3 | 0.1758 |
| GRU+Att | 9.28e-3 | 3.94e-3 | 2.56e-2 | 5.04e-3 | 3.45e-2 | 5.85e-3 | 0.1539 |
| FineNet | 1.34e-2 | 4.18e-3 | 2.83e-2 | 6.14e-3 | 3.85e-2 | 6.45e-3 | 0.1905 |
| RankLSTM | 1.81e-2 | 4.35e-3 | **3.43e-2** | **6.78e-3** | 4.16e-2 | 7.04e-3 | 0.2318 |
| **FinGAT** | **2.03e-2** | **4.63e-3** | 3.01e-2 | 6.24e-3 | **4.57e-2** | **7.15e-3** | **0.2579** |
| Improv. | 12.15% | 6.43% | -12.22% | -7.96% | 9.85% | 1.56% | 11.25% |

**Observations:**
- FinGAT significantly outperforms all competing methods on three datasets, especially on MRR and Precision.
- For top-5 (K=5), FinGAT consistently leads: average improvement 12.23% on MRR and 7.93% on Precision against RankLSTM.
- Top-K with smaller K is more useful for users [18], [30].
- Confirms effectiveness of modeling stock-stock, stock-sector, and sector-sector interactions.
- Outperforming RankLSTM verifies that *learning latent relationships* > using pre-defined relations.
- ACC improvement is minor — slight weakness in stock movement prediction.

#### Evaluation without Sector Info

Sector information is not always accessible. We devise **FinGAT-NT** (No Tier):

1. **Remove** sector-level modeling (Section 4.2).
2. **Remove** intra-sector graphs; instead create a single fully-connected graph $G_T$ of *all* stocks. Initial node feature: $\mathbf{a}^{s_q}_i$.
3. After GAT and long-term sequential learning, replace Eq. 13 with:

$$\pi^F_i(s_q) = \text{ReLU}([\pi^G_i(s_q) \| \pi^A_i(s_q)] W_f) \tag{19}$$

Same model learning as Section 4.3.

FinGAT-NT has high computation as #stocks grows, so we suggest selecting subsets. We sort all Taiwan Stock stocks by **market value** and create five subsets:
- **Best 10:** 10 highest-market-value stocks
- **Worst 10:** 10 lowest
- **Best 5 Worst 5:** 5 highest + 5 lowest
- **Random 10:** randomly select 10
- **Uniform 10:** divide into 10 zones, randomly pick 1 per zone

> **Figure 3:** Results on recommendation without sector information in terms of MRR@3 for different data subsets of Taiwan Stock dataset. (Bar chart comparing MLP, GRU, GRU+Att, FineNet, Rank-LSTM, FinGAT-NT across 5 subsets; Y-axis MRR@3 ∈ [0.8, 1.0].)

**Findings:**
- FinGAT-NT outperforms competing methods on all subsets.
- Superiority is more apparent on "Best 10", "Worst 10", "Best 5 Worst 5" — *latent relationships between homogeneous (similar market value) stocks are stronger.*

#### Ablation Study

Submodels of FinGAT (last three remove one component):
1. **Full Model:** all components.
2. **w/o intra:** remove $\tau^G_i(s_q)$ from intra-sector GAT.
3. **w/o inter:** remove $\tau_i(\pi_c)$ from inter-sector GAT.
4. **w/o MTL:** optimize only on pairwise ranking loss.
5. **w/ MSE:** replace movement (BCE) loss with MSE; sigmoid in Eq. 14 also removed.

**Table 3: Results on ablation study.**

##### Taiwan Stock

| Model | MRR@5 | Prec@5 | MRR@10 | Prec@10 | MRR@20 | Prec@20 | ACC |
|-------|------:|-------:|-------:|--------:|-------:|--------:|----:|
| **Full model** | **0.4391** | **0.1133** | **0.8479** | **0.2022** | **1.4106** | **0.2756** | **0.5682** |
| w/o intra | 0.3576 | 0.1033 | 0.7128 | 0.1317 | 1.3406 | 0.2586 | 0.5412 |
| w/o inter | 0.3950 | 0.1122 | 0.7464 | 0.1511 | 1.3887 | 0.2611 | 0.5509 |
| w/o MTL | 0.4215 | 0.1127 | 0.8023 | 0.1856 | 1.4089 | 0.2723 | 0.5342 |
| w/ MSE | 0.3486 | 0.0744 | 0.6867 | 0.1228 | 1.1783 | 0.1850 | 0.5078 |

##### S&P 500

| Model | MRR@5 | Prec@5 | MRR@10 | Prec@10 | MRR@20 | Prec@20 | ACC |
|-------|------:|-------:|-------:|--------:|-------:|--------:|----:|
| **Full model** | **0.1974** | **0.0419** | **0.3357** | **0.0677** | **0.5687** | **0.0989** | **0.5425** |
| w/o intra | 0.1432 | 0.0355 | 0.2391 | 0.0500 | 0.4282 | 0.0793 | 0.5284 |
| w/o inter | 0.1369 | 0.0301 | 0.2382 | 0.0398 | 0.4115 | 0.0877 | 0.5371 |
| w/o MTL | 0.1773 | 0.0409 | 0.2904 | 0.0581 | 0.5033 | 0.0892 | 0.5411 |
| w/ MSE | 0.1072 | 0.0172 | 0.2203 | 0.0387 | 0.3808 | 0.0683 | 0.5177 |

**Findings:**
- Full FinGAT is the best across all metrics — every component contributes.
- **Removing intra-sector → biggest drop:** latent stock relations have direct, significant impact.
- Pairwise ranking loss alone (w/o MTL) → smallest drop, but joint movement prediction still helps.
- **w/ MSE drastically hurts performance** — squared loss is flatter than BCE → harder optimization.

### 5.3 Hyperparameter Analysis

> **Figure 4:** Performance by varying (a)&(b) number of training weeks; (c)&(d) embedding size; (e)&(f) balance weight $\delta$.

#### Number of Training Weeks

Vary number of past weeks $\in \{1, 2, 3, 4\}$.
- **At least 3 weeks** → higher MRR and precision.
- 1-2 weeks → worse performance (missing long-term info).
- Confirms the need for both short- and long-term trends.

#### Embedding Size

Vary embedding dim $\in \{8, 16, 32, 64\}$.
- **Best at 16.**
- Too small (8) → underfitting.
- Too large (32, 64) → overfitting.
- Suggestion: use 16.

#### Balancing Parameter δ

Vary $\delta \in \{0, 0.0001, 0.001, 0.01, 0.1, 1\}$.
- **Best at δ = 0.01.**
- Small δ → less BCE contribution, more ranking loss.
- $\delta = 0$ (rank only) or $\delta = 1$ (move only) → much worse.
- Suggestion: $\delta = 0.01$.

### 5.4 Exploring Latent Interactions via Attention Weights

#### Intra-sector Attention

> **Figure 5:** Visualization of attention weights between same-sector stocks. (a) "Consumer & Goods" sector in Taiwan stock data. (b) "Energy" sector in S&P 500 data.

- **Figure 5(a) "Consumer & Goods":** High-attention cells form **subgroups** (stock IDs 0-3, 10-12). Reflects supply-chain collaboration among manufacturers, retailers, distributors [25].
- **Figure 5(b) "Energy":** Attention weights are **uniformly distributed** but low — interactions are significant due to scarcity-driven competition (e.g., oil price war) and high fluctuation [3].

Sector-sector attention provides insights for investor decision-making.

#### Inter-sector Attention

> **Figure 6:** Visualization of inter-sector attention weights. (a) Taiwan Stock — sectors: Semiconductors, Construction, Biotechnology & Medicine, Consumer Goods, Financial Insurance. (b) S&P 500 — sectors: Materials, Energy, Financials, Industrials, Information Technology, Consumer Staples, Communication Services, Health Care, Consumer Discretionary.

- **Taiwan (Fig 6a):** "Semiconductors" ↔ "Construction" stand out (both fundamentals of Taiwan's economy). "Consumer Goods" ↔ "Financial Insurance" also significant (supply/demand via financial behaviors).
- **S&P 500 (Fig 6b):** "Energy" sector has higher correlation with various sectors — oil/gasoline/fossil fuels critically impact the market [22], [24].
- FinGAT learned these without prior knowledge.

#### Distributions of Attention Weights

> **Figure 7:** Distributions and variances of attention weights (Taiwan Stock).

- **Fig 7(a) — Distribution of attention weights:**
  - **Inter-sector (blue):** right-skewed → only a few sectors are highly correlated. Market dominated by minority leading sectors (e.g., "Semiconductor").
  - **Intra-sector (orange):** concentrated in (0.2, 0.4) → no overly influential stocks within any sector.
- **Fig 7(b) — Variance of attention weights across test instances:**
  - **Inter-sector:** higher variance — leading sectors drastically influence others.
  - **Intra-sector:** lower variance — within-sector dependencies are stable.

---

## 6. Conclusion

This work aims at recommending the most profitable stocks using price time series and sector information. We develop **FinGAT**, with three novelties:

1. **No pre-defined relationships required** — exploit GAT to automatically learn latent interactions between stocks and sectors.
2. **Two-level hierarchy** (stocks ↔ sectors) for both fine-grained and coarse-grained relationship modeling.
3. **Multi-task objective** jointly optimizing profitable stock recommendation and stock movement prediction.

Experiments on Taiwan Stock, S&P 500, and NASDAQ show FinGAT significantly outperforms SOTA baselines and works even without sector information.

**Future Extensions:**

1. **Joint graph structure inference** — currently fully-connected; learn better graph structure.
2. **Knowledge graph from company metadata** — encode finer-grained correlations.
3. **News-based representation** — incorporate stock-related news for further improvement.

---

## Acknowledgments

Supported by Ministry of Science and Technology (MOST) of Taiwan under grants 109-2636-E-006-017 (MOST Young Scholar Fellowship) and 109-2221-E-006-173, and by Academia Sinica under grant AS-TP-107-M05.

---

## References

[1] Yujin Baek and Ha Young Kim. *Modaugnet: A new forecasting framework for stock market index value with an overfitting prevention LSTM module and a prediction LSTM module.* Expert Systems with Applications, 113:457–480, 2018.

[2] George EP Box, Gwilym M Jenkins, Gregory C Reinsel, Greta M Ljung. *Time series analysis: forecasting and control.* 2015.

[3] David C Broadstock, Ying Fan, Qiang Ji, Dayong Zhang. *Shocks and stocks: a bottom-up assessment of the relationship between oil prices, gasoline prices and the returns of Chinese firms.* The Energy Journal, 37, 2016.

[4] Li-Juan Cao, Francis Eng Hock Tay. *Support vector machine with adaptive parameters in financial time series forecasting.* IEEE Transactions on Neural Networks, 14(6):1506–1518, 2003.

[5] Kai Chen, Yi Zhou, Fangyan Dai. *A LSTM-based method for stock returns prediction: A case study of China stock market.* IEEE Big Data, 2823–2824, 2015.

[6] Yingmei Chen, Zhongyu Wei, Xuanjing Huang. *Incorporating corporation relationship via graph convolutional neural networks for stock price prediction.* CIKM, 1655–1658, 2018.

[7] Kyunghyun Cho et al. *Learning phrase representations using RNN encoder–decoder for statistical machine translation.* EMNLP, 1724–1734, 2014.

[8] Minsu Cho, Jian Sun, Olivier Duchenne, Jean Ponce. *Finding matches in a haystack: A max-pooling strategy for graph matching in the presence of outliers.* CVPR, 2083–2090, 2014.

[9] Bhuwan Dhingra, Hanxiao Liu, Zhilin Yang, William Cohen, Ruslan Salakhutdinov. *Gated-attention readers for text comprehension.* ACL, 1832–1846, 2017.

[10] Daizong Ding, Mi Zhang, Xudong Pan, Min Yang, Xiangnan He. *Modeling extreme events in time series prediction.* KDD, 1114–1122, 2019.

[11] Robert F Engle. *Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation.* Econometrica, 987–1007, 1982.

[12] Fuli Feng, Huimin Chen, Xiangnan He, Ji Ding, Maosong Sun, Tat-Seng Chua. *Enhancing stock movement prediction with adversarial training.* IJCAI-19, 5843–5849, 2019.

[13] Fuli Feng, Xiangnan He, Xiang Wang, Cheng Luo, Yiqun Liu, Tat-Seng Chua. *Temporal relational ranking for stock prediction.* TOIS, 37(2):1–30, 2019.

[14] Jie Feng, Yong Li, Chao Zhang, Funing Sun, Fanchao Meng, Ang Guo, Depeng Jin. *DeepMove: Predicting human mobility with attentional recurrent networks.* WWW, 1459–1468, 2018.

[15] Raehyun Kim, Chan Ho So, Minbyul Jeong, Sanghoon Lee, Jinkyu Kim, Jaewoo Kang. *HATS: A hierarchical graph attention network for stock movement prediction.* arXiv:1908.07999, 2019.

[16] Diederik P Kingma, Jimmy Ba. *Adam: A method for stochastic optimization.* ICLR, 2015.

[17] Junhyun Lee, Inyeop Lee, Jaewoo Kang. *Self-attention graph pooling.* ICML, 3734–3743, 2019.

[18] Dong Li, Ruoming Jin, Jing Gao, Zhi Liu. *On sampling top-K recommendation evaluation.* KDD '20, 2114–2124, 2020.

[19] Hao Li, Yanyan Shen, Yanmin Zhu. *Stock price prediction using attention-based multi-input LSTM.* ACML, 454–469, 2018.

[20] Thang Luong, Hieu Pham, Christopher D. Manning. *Effective approaches to attention-based neural machine translation.* EMNLP, 1412–1421, 2015.

[21] Daiki Matsunaga, Toyotaro Suzumura, Toshihiro Takahashi. *Exploring graph neural networks for stock market predictions with rolling window analysis.* NeurIPS 2019 Workshop on Robust AI in Financial Services, 2019.

[22] Dinh Hoang Bach Phan, Susan Sunila Sharma, Paresh Kumar Narayan. *Stock return forecasting: some new evidence.* International Review of Financial Analysis, 40:38–51, 2015.

[23] Omer Berat Sezer, Mehmet Ugur Gudelek, Ahmet Murat Ozbayoglu. *Financial time series forecasting with deep learning: A systematic literature review: 2005–2019.* Applied Soft Computing, 90:106181, 2020.

[24] Wesley D Sine, Brandon H Lee. *Tilting at windmills? The environmental movement and the emergence of the US wind energy sector.* Administrative Science Quarterly, 54(1):123–155, 2009.

[25] Hartmut Stadtler, Christoph Kilger. *Supply chain management and advanced planning, vol. 4.* Springer, 2002.

[26] Jiexiong Tang, Chenwei Deng, Guang-Bin Huang. *Extreme learning machine for multilayer perceptron.* IEEE TNNLS, 27(4):809–821, 2015.

[27] Yu-Che Tsai et al. *FineNet: a joint convolutional and recurrent neural network model to forecast and recommend anomalous financial items.* RecSys, 536–537, 2019.

[28] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, Yoshua Bengio. *Graph attention networks.* ICLR, 2018.

[29] Liheng Zhang, Charu Aggarwal, Guo-Jun Qi. *Stock price prediction via discovering multi-frequency trading patterns.* KDD, 2141–2149, 2017.

[30] Shuai Zhang, Lina Yao, Aixin Sun, Yi Tay. *Deep learning based recommender system: A survey and new perspectives.* ACM Comput. Surv., 52(1), 2019.

[31] Yue-Jun Zhang, Yi-Ming Wei. *The crude oil market and the gold market: Evidence for cointegration, causality and price discovery.* Resources Policy, 35(3):168–177, 2010.

---

## Author Biographies

**Yi-Ling Hsu** is a graduate student in the Master Program in Statistics, National Taiwan University (NTU), Taipei, Taiwan. She received a Bachelor's degree in Statistical Science from National Cheng Kung University (NCKU), Tainan, Taiwan, in 2020. She is also a research assistant in Networked Artificial Intelligence Laboratory at NCKU. Her research interests: Data Mining, Machine Learning, Financial Technology, Statistics.

**Yu-Che Tsai** is a Master graduate student in the Department of Computer Science and Information Engineering, NTU, Taipei, Taiwan. He received a Bachelor's degree in Statistical Science from NCKU in 2020. He is also a research assistant in NetAI Lab at NCKU. Research interests: Data Mining, Machine Learning, Recommender Systems, Deep Learning. Papers published in KDD 2020, CIKM 2019, RecSys 2019.

**Cheng-Te Li** is an Associate Professor at the Institute of Data Science and Department of Statistics, NCKU, Tainan, Taiwan. He received his Ph.D. degree (2013) from Graduate Institute of Networking and Multimedia, NTU. Before NCKU, he was an Assistant Research Fellow (2014–2016) at CITI, Academia Sinica. Research: Machine Learning, Deep Learning, Data Mining, Social Networks/Media Analysis, Recommender Systems, NLP. Papers at KDD, WWW, ICDM, CIKM, SIGIR, IJCAI, ACL, EMNLP, NAACL, RecSys, ACM-MM. Leads NetAI Lab at NCKU.
