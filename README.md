# Understanding Self-Attention in Transformers

This document provides a supplementary explanation of self-attention, inspired by the explanation around **46:50** in the following video:

📺 [Bilibili Video Link](https://www.bilibili.com/video/BV1xoJwzDESD/?spm_id_from=333.337.search-card.all.click&vd_source=594f8fd1e28e6148964bda737696c684)

---

## 🧠 Key Differences: Self-Attention vs Traditional Attention

- **Traditional Attention** (e.g., in translation tasks):
  - Driven by the **output**, attending to the **input**.
  - Example:
    - Input: `"我爱水课"`
    - Output: `"I love easy courses"`
    - Each word in the output sequence determines which parts of the input sequence to attend to, forming an **output-oriented attention** mechanism.

- **Self-Attention**:
  - Words within the **same input sequence attend to each other**.
  - This mechanism captures intra-sentence dependencies by letting every token relate to every other token.
  - Result: A **semantically enriched input matrix** using Q (Query), K (Key), and V (Value).
  - This understanding is mainly applied on the **Encoder side**, where self-attention helps to model internal relationships.

---

## 🔄 Decoder-Side: Masked Self-Attention + Cross-Attention

### Input Format with Teacher Forcing

During training, we use the **teacher forcing** strategy where the decoder receives the ground truth output shifted to the right, with a `<start>` token prepended.

For example, the target sentence `"I love easy courses"` is transformed into:

`<start> I love easy courses`

The full training target becomes:

`["I", "love", "easy", "courses", "<end>"]`

- We **prepend** `<start>` to the decoder input.
- We **remove** `<end>` from the training input sequence (it remains in the target output).

### Masked Self-Attention

- Q and K are generated from this decoder input, resulting in a `[5 × 512]` matrix.
- To ensure causality (i.e., no looking ahead), we apply **lower triangular masking**:
  - The upper triangular portion of the attention matrix is filled with `-inf`, so after softmax, it becomes zero.
- The resulting masked attention matrix of shape `[5 × 512]` is passed on.

---

## 🎯 Cross-Attention in the Decoder

- After masked self-attention, the next step is **cross-attention**, which resembles traditional attention.
- Here, the decoder attends to the encoder outputs to enrich its intermediate state.
- Unlike traditional models (often **single-head**), Transformer uses **multi-head cross-attention**.

---

## 🧮 Computation: Cross-Attention Step-by-Step

### Encoder Output

Suppose the encoder processes the input:

`["我", "爱", "水", "课"]`
 
Resulting in a matrix of shape: `[4 × 512]`

### Decoder State (from Masked Self-Attention)

For the decoder input:

`<start>, I, love, easy, courses`

The masked self-attention produces a state matrix of: `[5 × 512]`

Each row corresponds to one decoding step, progressively revealing more context:

| Row | Target Prediction | Attended Tokens |
|-----|-------------------|------------------|
| 1   | I                 | `<start>` |
| 2   | love              | `<start>`, I |
| 3   | easy              | `<start>`, I, love |
| 4   | courses           | `<start>`, I, love, easy |
| 5   | <end>             | `<start>`, I, love, easy, courses |

---

### Step 1: Generate Q, K, V

```text
Q = Decoder state × W_Q     = [5 × 512] × [512 × 512] → [5 × 512]
K = Encoder output × W_K    = [4 × 512] × [512 × 512] → [4 × 512]
V = Encoder output × W_V    = [4 × 512] × [512 × 512] → [4 × 512]

Scores    = Q × K^T          = [5 × 512] × [512 × 4] → [5 × 4]
Weights   = softmax(scores) = [5 × 4]
Output    = Weights × V      = [5 × 4] × [4 × 512] → [5 × 512]
```

The result is a `[5 × 512]` matrix that is then passed into the next feed-forward layers.

---

## ❓ Why No `<end>` in Encoder Input?

- **Encoder**: Processes input **in parallel**, inherently knows boundaries.
- **Decoder**: Generates output **step-by-step**, requires:
  - `<start>` to begin generation.
  - `<end>` to terminate generation.

---

## 🙏 Final Notes

This document represents my **personal understanding** of the self-attention and decoder attention mechanisms in Transformers.

If there are any mistakes or inaccuracies, your **corrections are highly appreciated**. Thank you!
