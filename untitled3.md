Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) are both evaluation metrics used to assess the performance of ranking systems, such as search engines and recommendation systems. Each metric has its own strengths and is suited for different scenarios. Here's a comparative overview:

---

### 🔍 Mean Reciprocal Rank (MRR)

**Definition:**MRR measures how quickly a system returns the first relevant item in a ranked list. It is calculated as the average of the reciprocal ranks of the first relevant item across all queries

**Formula:**
$$
\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}
$$
\where \(N\) is the number of queries, and \(\text{rank}_i\) is the position of the first relevant item for the \(i\)-th query

**Pros:**
-Simple to compute and interpret
-Emphasizes the importance of retrieving the first relevant item quickly
-Ideal for tasks where users expect a single correct answer promptly, such as question-answering systems

**Cons:**
-Ignores the relevance of items beyond the first relevant one
-Does not account for multiple relevant items in the list
-May not be suitable for applications where users are interested in exploring a list of relevant items

---

### 📊 Normalized Discounted Cumulative Gain (NDCG)

**Definition:* NDCG evaluates the ranking quality by considering the position of relevant items and their graded relevance. It discounts the gain of each relevant item based on its position in the lis.

**Formula:*
$$
\text{DCG}_K = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
$$

$$
\text{NDCG}_K = \frac{\text{DCG}_K}{\text{IDCG}_K}
$$
where \(\text{rel}_i\) is the relevance score of the \(i\)-th item, and \(\text{IDCG}_K\) is the ideal DCG for the top \(K\) item.

**Pros:**
 Considers the position of all relevant items in the lis.
 Incorporates graded relevance, allowing for varying levels of relevanc.
 Provides a more comprehensive evaluation of ranking qualit.

**Cons:**
 More complex to compute compared to MR.
 Requires predefined relevance scores for item.
 Normalization can be sensitive to the choice of \(K\.

---

### 🔄 Comparative Overvie

| Feature                 | MRR                               | NDCG                                  |
|-------------------------|-----------------------------------|---------------------------------------|
| Focus                   | First relevant item               | Overall ranking quality               |
| Relevance Grading       | Binary (relevant or not)          | Graded relevance levels               |
| Position Sensitivity    | Only the first relevant item      | All relevant items, discounted by position |
| Ideal Use Case          | Tasks requiring quick retrieval of a single relevant item | Tasks where the order of multiple relevant items matter |

---

### ✅ When to Use Each Metric

- **Use MRR** when:
 - The primary goal is to retrieve the first relevant item quicky.
 - The application involves tasks like question answering or fact retrievl.

- **Use NDCG** when:
 - The ranking of multiple relevant items is importat.
 - The application involves tasks like search engines or recommendation systems where users may explore a list of relevant ites.

--

In summary, MRR is straightforward and emphasizes the importance of the first relevant item, making it suitable for applications with a single correct answer. NDCG, on the other hand, provides a more nuanced evaluation by considering the position and relevance of all relevant items, making it ideal for tasks involving ranked lists of ites.

--- 