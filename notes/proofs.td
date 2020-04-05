[define:I,\mathbb{I}]
[define:Q,\mathbb{Q}]
[define:Z,\mathbb{Z}]
[define:C,\mathbb{C}]
[define:F,\mathbb{F}]
[define:R,\mathbb{R}]
[define:N,\mathbb{N}]
[define:unif,\mathcal{U}]
[define:Tor,\operatorname{Tor}]
[define:ra,\rightarrow]
[define:rr,\Rightarrow]
[define:llrr,\Leftrightarrow]
[define:subq,\subseteq]
[define:sub,\subset]
[define:nsubq,\not\subseteq]
[define:nsub,\not\subset]
[define:sep,\hbox{ }]
[define:inv,^{\raisebox{.2ex,$\scriptscriptstyle-1$}}]
[define:thnew,^{\text{th}}]
[define:ifnew,\text{ if }]
[define:elsenew,\text{ else }]
[define:for,\text{ for }]

# Expected value of kth element of an ordered sequence of elements from some interval

## Question:

Let each ordered sequence of $n$ unique elements $X_1, \dots, X_n$ from the interval $(a, b)$ be equally probable. What is the expected value of the element $X_k$?

## Answer:

Consider random variables $Y_1, \dots, Y_n$ sampled from a uniform  distribution over $(a, b)$ such that no two $Y_i$ are equal. First, we show that any ordered sequence made from $Y_1, \dots, Y_n$ is equally probable.

Let $X_1 = Y_{i_1}, \dots, X_n = Y_{i_n}$ be the ordered sequence made from our random variables $Y_1, \dots, Y_n$. Then we see the probability density function, $f$, at a specific random sequence is

$$f_X([X_1, \dots, X_n]) = \sum_{Y_1, \dots, Y_n \in (a, b)^n; Y_i \text{ unique}} P(Y_1, \dots, Y_n) $$
$$= \sum_{Y_1, \dots, Y_n \in (a, b)^n; Y_i \text{ unique}} f_{Y_1}(Y_1)\cdots f_{Y_n}(Y_n)$$

Note that the value of a uniform probability density does not change if we introduce a finite number of holes, thus:

$$$
f_{Y_i}(Y_i) = 
\begin{cases}
1,  & \text{if $Y_i \in (a, b)$ and $Y_i \neq Y_j$ for $j \neq i$} \\
0, & \text{otherwise}
\end{cases}
$$$

To see this 