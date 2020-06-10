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

$$f_X([X_1, \dots, X_n]) = \sum_{Y_1, \dots, Y_n \in \text{ Permuations}(X_1, \dots, X_n)} f_{Y_1, \dots, Y_n}(Y_1, \dots, Y_n) $$
$$= \sum_{Y_1, \dots, Y_n \in \text{ Permuations}(X_1, \dots, X_n)} f_{Y_1}(Y_1)\cdots f_{Y_n}(Y_n)$$

Note that the value of a uniform probability density does not change if we introduce a finite number of holes (as the integral over the pdf does not change in value), thus:

$$
f_{Y_i}(Y_i) = 
\begin{cases}
\frac{1}{b-a},  & \text{if $Y_i \in (a, b)$ and $Y_i \neq Y_j$ for $j \neq i$} \\
0, & \text{otherwise}
\end{cases}
$$

**NOTE**: *I think I can rewrite it this way, because all I'm doing is reorganizing the points of the composite $Y$ pdf into groups that produce the same ordered sequence. So when I integrate over the pdf of the ordered sequences, it's integrating over the same region as the composite $Y$ pdf.* **But the multiplier of $n!$ can't be right or the pdf for $f_X$ will integrate to greater than 1!**

Therefore we can rewrite $f_X$:

$$f_X([X_1, \dots, X_n]) = \sum_{Y_1, \dots, Y_n \in \text{ Permuations}(X_1, \dots, X_n)}  \frac{1}{(b-a)^n}$$
$$=\frac{n!}{(b-a)^n}$$
