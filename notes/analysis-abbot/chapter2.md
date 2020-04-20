# Sequences and Series

## 2.2 The Limit of a Sequence

### Notes

**Definition**: A *sequence* is a function whose domain is $\N$.

- This is really just an ordered list where the function $f(n)$ indexes the list at the $n$th entry.

**Definition (Convergence of a Sequence)** A sequence $(a_n)$ *converges* to a real number $a$ if $\forall \epsilon > 0, \exists N \in \N$ s.t. if $n \geq N$, then $|a - a_n| < \epsilon$.

**Definition**: Given $a \in \R$ and $\epsilon > 0$, the set

$$V_\epsilon(a) = \{x \in \R | (|x - a| < \epsilon\}$$

is the $\epsilon$-neighborhood of $a$

**Definition (Convergence of a sequence: Topological Version)**: A sequence $(a_n)$ converges to $a$ if, given any $\epsilon$-neighborhood $V_\epsilon(a)$ of $a$, $\exists N \in \N$ such that for all $n \geq N$, $a_n \in V_\epsilon(a)$.

**Example**: Consider $(a_n)$, where $a_n = \frac{1}{\sqrt{n}}$. Prove that $\lim (a_n) = 0$
*Proof*: Consider arbitrary $\epsilon > 0$. Note that
$$
|a_n - a| = |a_n - 0| = |a_n| = |\frac{1}{\sqrt{n}}| = \frac{1}{\sqrt{n}}
$$
Choose $N > \frac{1}{\epsilon^2}$. Note that 
$$ n \geq N > \frac{1}{\epsilon^2} \Rightarrow \frac{1}{n} \leq \frac{1}{N} \leq \epsilon^2 \Rightarrow \frac{1}{\sqrt{n}} \leq \frac{1}{\sqrt{N}} \leq \epsilon $$

Since $a_n = |a_n - 0|$, then
$$ |a_n - 0| \leq \epsilon $$
$\blacksquare$l

**Example**: Show that
$$ \lim \Big( \frac{n+1}{n}\Big) = 1$$
*Pre-proof*: Consider
$$ \Big| \frac{n+1}{n} - 1 \Big| < \epsilon .$$
Note that 
$$ \Big| \frac{n+1}{n} - 1 \Big| = \Big| \frac{n+1}{n} - \frac{n}{n} \Big| = \Big| \frac{1}{n} \Big| $$
$$ =\frac{1}{n} .$$

Thus
$$ \Big| \frac{n+1}{n} - 1 \Big| = \frac{1}{n} < \epsilon .$$
$$ \Rightarrow n > \frac{1}{\epsilon} $$
*Proof*: Choose $N \in \N$ such that $N > \frac{1}{\epsilon}$. Then for $n \geq N$ this means
$$ n \geq N > \frac{1}{\epsilon} $$
$$ \Rightarrow \frac{1}{n} \leq \frac{1}{N} < \epsilon $$
$$ \Rightarrow \Big| \frac{n+1}{n} - 1 \Big| < \epsilon $$
$\blacksquare$

**My Corollary 1:** Consider sequence $(a_n)$. $(a_n)$ converges to $a$ if and only if $|a_n - a| \rightarrow 0$.
*Proof*:

1. ($\Rightarrow$): We know $(a_n) \rightarrow a$. Thus for arbitrary $\epsilon > 0$, $\exists N \in \N$ s.t. $\forall n \geq N$,
$$ |a_n - a| < \epsilon . $$
Denote the sequence $|a_n - a| = d_n$. Consider arbitrary $\epsilon' > 0$. Note that
$$ |d_n - 0| = d_n .$$
Thus, to show $\lim d_n = 0$, we simply show that $\exists N' \in \N$ s.t. $\forall n \geq N'$,
$$ d_n < \epsilon' .$$
Choose $N' = N$, where $\forall n \geq N$, $|a_n - a| < \epsilon'$. Therefore,
$|a_n - a| = d_n < \epsilon'$

2. ($\Leftarrow$): We know $(d_n = |a_n - a|) \rightarrow 0$. Thus, for arbitrary $\epsilon > 0$, $\exists N \in \N$ s.t. $\forall n \geq N$,
$$ |a_n - a| = d_n < \epsilon . $$
The above statement shows, for the same $N$, by the equality between $|a_n - a| = d_n$, 
$$ |a_n - a| < \epsilon .$$
Therefore, $(a_n)$ converges to $a$. $\blacksquare$

**Definition**: A sequence that does not converge is said to *diverge*.

### Exercises

**2.1.b**: Prove that $\lim \frac{3n + 1}{2n + 5} = \frac{3}{2}$. 

*Proof*: First consider the inequality we want to show:
$$ |\frac{3n + 1}{2n + 5} - \frac{3}{2} | < \epsilon$$
Note that
$$ |\frac{3n + 1}{2n + 5} - \frac{3}{2} | = |\frac{6n + 2}{4n + 10} - \frac{6n + 15}{4n + 10} |$$
$$ = |\frac{-13}{4n + 10} | = \frac{13}{4n + 10} $$
So now solving for the relationship between $n$ and $\epsilon$ in terms of $\epsilon$.
$$ \frac{13}{4n + 10} < \epsilon \Rightarrow \frac{13}{4\epsilon} - \frac{5}{2} < n $$
Thus we choose $N > \frac{13}{4\epsilon} - \frac{5}{2}$. As $n \geq N \Rightarrow n > \frac{13}{4\epsilon} - \frac{5}{2}$, by the above algebra, we can conclude
$$ |\frac{3n + 1}{2n + 5} - \frac{3}{2} | < \epsilon$$
$\blacksquare$


**2.2**: What happens if we reverse the order of the quantifiers in our definition of convergence:
*Definition*: A sequence $(x_n)$ *verconges* to $x$ if $ \exists \epsilon > 0$ such that $\forall N \in \N$ it is true that $n \geq N$ implies $|x_n -x| < \epsilon$ 
Give an example of a vercongent sequence that is divergent. What exactly is being described.

*Solution*: When we reverse the order of the quantifiers, our definition states we only need to find one $\epsilon > 0$ such that $|x_n - x| < \epsilon$ for all elements of the sequence (every $n \in \N$ is $\geq$ some $N \in \N$). Thus our entire sequence is bounded by some $\epsilon > 0$. Note that if at some $N \in \N$, our subsequence becomes *vercongent*, we can change the $\epsilon$ to $\max (\epsilon, |x_1-x|, \dots, |x_{N-1}-x|)$ and thus it will remain *vercongent*. So when a sequence is *vercongent*, it's distance to $x$ is bounded. A diverging sequence that is *vercongent* is

$$ (1, -1, 1, -1, \dots) $$

It *verconges* to $0$ with $\epsilon = 2$.


**2.4**: Argue that the sequence

$$ 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, \text{(4 zeros)}, 1, \dots $$

does not converge to zero. For what values of $\epsilon > 0$ does there exists a response $N$ and for which is there no suitable response?

*Proof*: Consider $\epsilon = \frac{1}{2}$. As the number of zeros between two 1's grows by one after each 1 and there are a finite number of 1's in the first $N$ elements, at any $N \in \N$, there are a finite number of 0's before the first 1 after $N$. Thus $\exists n \geq N$ such that 

$$ |a_n - 0 | = |a_n| > \epsilon = \frac{1}{2} $$

namely where $a_n = 1$. $\blacksquare$
For any $\epsilon > 1$, all elements of the sequence are within $\epsilon$ distance of $0$.


**2.5**: Let $[[x]]$ be the greatest integer less than or equal to $x$. For example, $[[\pi]] = 3$ and $[[3]] = 3$. Find $\lim a_n$ and supply proofs for each conclusion if

(a) $a_n = [[1/n]]$

*Proof*: We will show that $\lim a_n = 1$. First consider the inequality for $\epsilon > 0$:

$$ |[[1/n]] - 1| < \epsilon $$

Note that as $1/n > 0$, but can be arbitrarily small for some large $n$, $[[1/n]] = 1$ for $n \geq 1$. Thus, in actuality, the sequence

$$ a_n = [[1/n]] $$

is just an infinite sequence of 1's. Thus, no matter which $\epsilon > 0$ we choose, $N = 1$ will suffice as for any $n \in \N$:

$$ |[[1/n]] - 1| = |1 - 1| = 0 < \epsilon .$$
$\blacksquare$

(b) $a_n = [[(10 + n)/2n]]$

*Proof*: We will show that $\lim a_n = 1$. First note that

$$ [[(10 + n)/2n]] = [[5/n + 1/2]] .$$

Similarly to part (a), as $5/n > 0$ for $n \in \N$, we see that 

$$ 5/n + 1/2 > 1/2 $$

However, for $n \geq 10$ we have

$$ 1/2 < 5/n + 1/2 \leq 1 .$$

Thus for all $n \geq 10$, $[[5/n + 1/2]] = 1$. Thus $N = 10$ is a suitable response for all $\epsilon > 0$.


Reflecting on the inverse relationship between the size of $\epsilon$ and the size of $N$, in this case, there is an $N$ that satisfies all $\epsilon$ since the series both repeat the same element after some $N \in \N$.

**6**:

**7**:

**8**:

## 2.3 The Algebraic and Order Limit Theorems

test changes