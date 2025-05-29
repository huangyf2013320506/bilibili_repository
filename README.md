# Understanding Self-Attention in Transformers

This document provides a supplementary explanation of self-attention, inspired by the explanation around **46:50** in the following video:

📺 [Bilibili Video Link](https://www.bilibili.com/video/BV1xoJwzDESD/?spm_id_from=333.337.search-card.all.click&vd_source=594f8fd1e28e6148964bda737696c684)

---

## 🔍 Key Differences: Self-Attention vs Traditional Attention

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
| 5   | `<end>`           | `<start>`, I, love, easy, courses |

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

# 中文版本 Chinese Version:
# Transformer 注意力机制个人理解补充

本项目是对Bilibili视频 [BV1xoJwzDESD](https://www.bilibili.com/video/BV1xoJwzDESD/?spm_id_from=333.337.search-card.all.click&vd_source=594f8fd1e28e6148964bda737696c684) 中大约 46:50 处关于自注意力机制解释的个人理解补充。

---

## 自注意力与传统注意力机制的区别

我个人理解，**自注意力（Self-Attention）**与传统的注意力机制主要区别在于其关注的焦点。

* **传统注意力机制**：以翻译任务为例（如“我爱水课”翻译成"I love easy courses"），传统注意力表现为**输出如何关注输入**。它是由输出驱动的，决定从输入中提取什么信息。可以理解为：**输出面向输入的关注**。

* **自注意力机制**：它**只关注自身内部**。在Encoder部分，自注意力是输入序列内部词与词之间的关系关注。它通过内部词汇间的相互解释和丰富，通过QKV矩阵得到一个语义更丰富的输入矩阵。可以理解为：**输入内部互相解释，丰富彼此的语义**。

    * **Encoder侧的自注意力**：输入序列的每个词都会与其他词计算注意力，从而捕获序列内部的依赖关系。

---

## Decoder 中的注意力机制

在Decoder部分，尤其是在 `teacher forcing` 的训练策略下，理解会有些不同：

### Masked Self-Attention (掩码自注意力)

在Decoder的第一层（Masked Self-Attention），输入的文本是 `"<start> I love easy courses"`（这是预期输出文本整体右移并在开头添加 `<start>` 信号，其原本完整的训练目标是 `["I", "love", "easy", "courses", "<end>"]`，也就是说我们移除了右侧的 `<end>` 并在左侧添加了 `<start>`）。

进行Q、K两个矩阵的线性变换后会得到一个 $5 \times 5$ 的注意力矩阵，然后会进行**下三角掩码**操作（将上三角注意力设置为负无穷大，这样经过 `softmax` 后那部分就会变成 $0$），最终得到一个 $5 \times 512$ 且经过“带掩码的注意力矩阵”处理过后的矩阵。这个矩阵将作为输入传递到下一个 **Multi-head Attention** 模块。

### Cross-Attention (交叉注意力)

Decoder中的第二个多头注意力模块不再是Encoder中的自注意力模块，而是**交叉注意力模块**。它更类似于传统的注意力机制（尽管区别在于Transformer的解码器中使用的是多头的交叉注意力机制，而传统的交叉注意力大多数是单头），它注重**Decoder的当前状态**（`"I love easy courses"`）和**Encoder的输出**（`“我爱水课”`）之间的关系。

#### 交叉注意力计算流程个人理解：

* **Encoder输出**： $[4 \times 512]$ （表示`["我","爱","水","课"]`的编码）
* **Decoder状态**： $[5 \times 512]$ （来自 Masked Self-Attention 的输出）
    这里的Decoder状态实际上是五行针对不同掩码情况时生成的 $512$ 维的行向量：
    * 第一行：只关注 `"<start>"`，用于预测 `"I"`。
    * 第二行：关注 `"<start>"` 和 `"I"`，用于预测 `"love"`。
    * 第三行：关注 `"<start>"`, `"I"`, `"love"`，用于预测 `"easy"`。
    * 第四行：关注 `"<start>"`, `"I"`, `"love"`, `"easy"`，用于预测 `"courses"`。
    * 第五行：关注 `"<start>"`, `"I"`, `"love"`, `"easy"`, `"courses"`，用于预测 `"<end>"`。

**步骤 1：生成 Q, K, V 矩阵**

* $Q = \text{Decoder状态} \times W_Q = [5 \times 512] \times [512 \times 512] = [5 \times 512]$
* $K = \text{Encoder输出} \times W_K = [4 \times 512] \times [512 \times 512] = [4 \times 512]$
* $V = \text{Encoder输出} \times W_V = [4 \times 512] \times [512 \times 512] = [4 \times 512]$

**步骤 2：计算注意力**

* $\text{注意力分数} = Q \times K^T = [5 \times 512] \times [512 \times 4] = [5 \times 4]$
* $\text{注意力权重} = \text{softmax}(\text{注意力分数}) = [5 \times 4]$
* $\text{输出} = \text{注意力权重} \times V = [5 \times 4] \times [4 \times 512] = [5 \times 512]$

---

## 为什么Encoder侧的输入不需要 `<end>` 标记？

* **Encoder**：它并行处理整个输入序列，天然地知道序列的边界。
* **Decoder**：它面向输出逐步生成，因此需要一个明确的停止信号 (`<end>`) 来指示何时停止生成。

---

**以上是我个人的理解，如有错误，还望大家斧正，非常感谢！**
