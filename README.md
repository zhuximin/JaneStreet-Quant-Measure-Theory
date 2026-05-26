### Theoretical Framework: A Measure-Theoretic Approach to Non-Parametric Feature Estimation

In high-frequency trading environments like Jane Street's markets, microscopic structural noise and extreme heavy-tailed distributions pose significant challenges. Traditional continuous parametric models often fail under such extreme skewness due to their sensitivity to severe outliers.

From the perspective of real analysis, we can model the market states on a probability space $(\Omega, \mathcal{F}, \mathbb{P})$, where the physical measure $\mathbb{P}$ encapsulates the complete information of market tail risks. The anonymized, highly volatile features in this dataset can be viewed as measurable functions $f: \Omega \to \mathbb{R}$.

Tree-based algorithms like the XGBoost model implemented in this baseline handle these features through recursive discrete partitioning. Mathematically, this splitting mechanism is equivalent to approximating the true feature distribution and expected returns using simple functions. We can rigorously express the model's output as:

$$\phi_n(x) = \sum_{k=1}^{n} c_k \chi_{E_k}(x)$$

where $\chi_{E_k}$ represents the indicator function corresponding to the tree's leaf node partition (a disjoint measurable set $E_k \in \mathcal{F}$), and $c_k$ is the predicted constant or leaf weight assigned to that specific partition.

Given the frequent stochastic jumps and extreme noise in financial data, calculating the direct expectation $\int_{\Omega} |f| d\mathbb{P}$ can result in massive variance or even divergence (i.e., the function may not be strictly Lebesgue integrable under extreme market conditions). By employing the simple function approximation $\phi_n(x)$ to partition the probability space, the tree model effectively performs a robust, non-parametric estimation under the true measure $\mathbb{P}$.
