

## ✅ **Summary (One-liner for Interview)**  
**RMSprop is an adaptive learning rate optimizer that speeds up convergence by dividing the gradient by a running average of its recent magnitudes.**

---

## 🔍 What is RMSprop?

**RMSprop = Root Mean Square Propagation**  
Designed to handle **non-stationary objectives**, especially effective in **online and mini-batch learning**.

### 🔧 Core Idea:
It keeps a **moving average of squared gradients** and divides the gradient by the root of this average to scale the learning rate **dynamically** per parameter.

---

## 📐 RMSprop Formula

Let:
- \( g_t \): current gradient
- \( E[g^2]_t \): running average of squared gradients
- \( \eta \): learning rate
- \( \gamma \): decay rate (usually ~0.9)
- \( \epsilon \): small number to avoid division by zero

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2
$$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t
$$

✅ **Key Insight**: It dampens the learning rate for **frequent parameters** and keeps it higher for **sparse updates**.

---

## ⚡ Why RMSprop Was Needed?

- **Standard SGD** uses the same learning rate for all parameters → can zigzag on steep slopes.
- **RMSprop** adapts learning rate **independently per parameter**.
- Helps in **faster convergence** in deep neural networks and **non-convex** loss surfaces.

---

## 📊 RMSprop vs Other Optimizers

| Optimizer   | Adaptive LR | Momentum | Works well when...                          | Notes                              |
|-------------|-------------|----------|---------------------------------------------|-------------------------------------|
| **SGD**     | ❌          | Optional | Large datasets, convex loss                 | Can be slow, needs tuning           |
| **Adagrad** | ✅          | ❌       | Sparse features (e.g., NLP)                 | LR keeps shrinking → early stop     |
| **RMSprop** | ✅ (EMA)    | ❌       | RNNs, time-series, deep models              | Fixes Adagrad’s LR decay issue      |
| **Adam**    | ✅          | ✅       | General purpose, very stable                | RMSprop + Momentum + Bias correction |
| **Adadelta**| ✅          | ❌       | Extension of RMSprop, no learning rate needed | Less used today                     |

---

## 🧠 RMSprop is Best For:
- Recurrent Neural Networks (RNNs)
- Time-series data
- Deep networks with noisy gradients or non-stationary objectives
- Training stability without fine-tuning LR

---

## 🧠 2-Liner Takeaway:
It improves upon Adagrad by preventing learning rate decay and works well in deep, non-stationary problems like RNNs.

