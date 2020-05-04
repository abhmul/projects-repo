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
$$ n \geq N > \frac{1}{\epsilon^2} \Rightarrow \frac{1}{n} \leq \frac{1}{N} < \epsilon^2 \Rightarrow \frac{1}{\sqrt{n}} \leq \frac{1}{\sqrt{N}} < \epsilon $$

Since $a_n = |a_n - 0|$, then
$$ |a_n - 0| < \epsilon $$
$\blacksquare$

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

**Definition 2.3.1**: A sequence $(x_n)$ is *bounded* if there exists a number $M > 0$ such that $|x_n| \leq M$ for all $n \in \N$. 

- Note that if a subseqeuence excluding the first $n$ elements is bounded by $M$, then the whole sequence is bounded. We can take the bound to be $\max (|x_1|,\dots, |x_n|, M)$.

**Theorem 2.3.2**: Every convergent sequence is bounded.

*Proof*: Name our convergent sequence $(x_n)$. By definition of convergence, $\forall \epsilon > 0$, $\exists N \in \N$ such that if $n >  N$ then $|x_n - x| < \epsilon$ for some $x \in \R$. Consider $\epsilon = 1$. Let $N \in \N$ be such that if $n > N$, $|x_n - x| < 1$. Note that

$$|x_n - x| < 1$$
$$\Rightarrow |x_n - x| + |x| < 1 + |x|$$
$$\Rightarrow |x_n - x + x| < 1 + |x|  \text{ (by triangle inequality) }$$
$$\Rightarrow |x_n| < 1 + |x|$$


Now consider

$$
M = \max (|x_1|, \dots, |x_N|, 1 + |x|)
$$

By construction, $M \geq x_i$ for all $i \in \N$. $\blacksquare$

**The below theorem shows that, given sequences and the fact that their individual limits exist, we can compute the limit of their algebraic compbinations intuitively.**

**Theorem 2.3.3 (Algebraic Limit Theorem)**: Let $\lim a_n = a$ and $\lim b_n = b$. Then,

1. $\lim (ca_n) = ca$, $\forall c \in \R$
2. $\lim (a_n + b_n) = a + b$
3. $\lim(a_nb_n) = ab$ 
4. $\lim(a_n / b_n) = a / b$

**Note:** Proofs for (3) and (4) on my notes are a little more complicated than in Abbot. See book for simpler proofs.

1. *Proof*: We know that $\lim(a_n) = a$. By definition of limit, this means that $\forall \epsilon > 0$, $\exists N \in \N$ s.t. $|a_n - a| < \epsilon$ for any $n > N$. Now consider the sequence $(ca_n)$. First, for the case $c \neq 0$, We will show that it converges to $ca$. Let $\epsilon' > 0$. Choose $N \in \N$ such that for $n > N$

$$|a_n - a| < \frac{\epsilon'}{|c|}$$
$$\Rightarrow |c||a_n - a| < \epsilon'$$
$$\Rightarrow |ca_n - ca| < \epsilon'$$

Thus for $n > N$, $ca_n \in (ca - \epsilon', ca + \epsilon')$. By definition of convergence, $ca_n \rightarrow ca$. 

Now consider the case where $c = 0$. This means $(ca_n)$ reduces to the sequence $(0, 0, 0, 0, \dots)$. To show this converges to $0$, consider $\epsilon' > 0$ and choose $N = 1$. Then consider for $n > N$:

$$|ca_n - 0| < \epsilon'$$
$$|0 - 0| < \epsilon'$$
$$0 < \epsilon'$$

Which is true by construction of $\epsilon'$. Thus the sequence $(0, 0, 0, 0, \dots)$ converges to $0$. $\blacksquare$

$\blacksquare$

2. *Proof*: We know that $\lim(a_n) = a$. By definition of limit, this means that $\forall \epsilon_a > 0$, $\exists N_a \in \N$ s.t. $|a_n - a| < \epsilon_a$ for any $n > N_a$. We also know that $\lim(b_n) = b$. By definition of limit, this means that $\forall \epsilon_b > 0$, $\exists N_b \in \N$ s.t. $|b_n - b| < \epsilon_b$ for any $n > N_b$. Let $\epsilon' > 0$ and $\epsilon_a = \epsilon_b = \frac{\epsilon'}{2}$ (this means $\epsilon' = \epsilon_a + \epsilon_b$). Choose $N = \max(N_a, N_b)$ such that for $n > N$:

$$|a_n - a| + |b_n - b| < \epsilon'$$
$$\Rightarrow |a_n - a + b_n - b| < \epsilon'$$
$$\Rightarrow |(a_n + b_n) - (a + b)| < \epsilon'$$

Thus for $n > N$, $a_n + b_n \in (a + b - \epsilon', a + b - \epsilon')$. By definition of convergence, $(a_n + b_n) \rightarrow a + b$. $\blacksquare$

3. *Proof*: We know that $\lim(a_n) = a$. By definition of limit, this means that $\forall \epsilon_a > 0$, $\exists N_a \in \N$ s.t. $|a_n - a| < \epsilon_a$ for any $n > N_a$. We also know that $\lim(b_n) = b$. By definition of limit, this means that $\forall \epsilon_b > 0$, $\exists N_b \in \N$ s.t. $|b_n - b| < \epsilon_b$ for any $n > N_b$. Let $\epsilon' > 0$ and choose

$$\epsilon_a =\frac{\epsilon'}{3b}$$
$$\epsilon_b = \min (\frac{\epsilon'}{3a}, \frac{\epsilon'}{3 \epsilon_a})$$

(that way $a\epsilon_b + b\epsilon_a + \epsilon_a \epsilon_b \leq \frac{\epsilon'}{3} + \frac{\epsilon'}{3} + \frac{\epsilon'}{3} = \epsilon'$).**We will handle the case where $a=0$ (and WLOG $b=0$) in exercise  2.3.7**. Choose $N = \max(N_a, N_b)$ such that for $n > N$:

$$b|a_n - a| + a|b_n - b| + |a_n - a||b_n - b| < \epsilon'$$
$$\Rightarrow |a_nb - ab| + |ab_n - ab| + |(a_n - a)b_n - (a_n - a)b| < \epsilon'$$
$$\Rightarrow |a_nb - ab + ab_n - ab + a_nb_n - ab_n - a_nb + ab| < \epsilon'$$
$$\Rightarrow |a_nb_n + a_nb - a_nb + ab_n - ab_n - ab - ab + ab| < \epsilon'$$
$$\Rightarrow |a_nb_n - ab| < \epsilon'$$

Thus for $n > N$, $a_nb_n \in (ab - \epsilon', ab + \epsilon')$. By definition of convergence, $(a_nb_n) \rightarrow ab$. $\blacksquare$

1. *Proof*: First we show that $\lim 1 / b_n \rightarrow 1 / b$. We know that $\lim(b_n) = b$. By definition of limit, this means that $\forall \epsilon_b > 0$, $\exists N_b \in \N$ s.t. $|b_n - b| < \epsilon_b$ for any $n > N_b$. Now consider

$$ |\frac{1}{b_n} - \frac{1}{b}|$$
$$= |\frac{b - b_n}{b_n b}|$$
$$\tag{1} = \frac{1}{|b||b_n|}|b - b_n|$$

Now choose $|b| > \epsilon_b > 0$ and let $n > N \in \N$ s.t. $|b - b_n| < \epsilon_b$. By definition of convergence, all $b_n \in (b - \epsilon_b, b + \epsilon_b)$ for $n > N$. This means we can construct the lower bound on $|b_n|$:

$$ 0 < |b| - \epsilon_b < |b_n| < |b| + \epsilon_b$$

Substituting into (1) we get

$$ |\frac{1}{b_n} - \frac{1}{b}| < \frac{1}{|b|(|b| - \epsilon_b)}\epsilon_b$$

Now choose $\epsilon' > 0$ and $\epsilon_b$ s.t. $\epsilon' = \frac{1}{|b|(|b| - \epsilon_b)}\epsilon_b$. To verify that a valid $\epsilon_b > 0$ exists for all choices of $\epsilon'$, we can solve the above equality for $\epsilon_b$

$$ \epsilon' = \frac{\epsilon_b}{|b|(|b| - \epsilon_b)}$$
$$ \Rightarrow |b|(|b| - \epsilon_b)\epsilon' = \epsilon_b$$
$$ \Rightarrow |b|^2\epsilon' - |b|\epsilon_b\epsilon' = \epsilon_b$$
$$ \Rightarrow |b|^2\epsilon' = \epsilon_b + |b|\epsilon_b\epsilon'$$
$$ \Rightarrow |b|^2\epsilon' = \epsilon_b(1 + |b|\epsilon')$$
$$ \Rightarrow \frac{|b|^2\epsilon'}{1 + |b|\epsilon'} = \epsilon_b$$

As all the terms on the left are positive, we're guaranteed to find an $\epsilon_b$ that works for any given $\epsilon' > 0$. Thus, for $n > N \in \N$ such that:

$$|b_n - b| <  \frac{|b|^2\epsilon'}{1 + |b|\epsilon'}$$

Our work above shows that

$$ |\frac{1}{b_n} - \frac{1}{b}| < \epsilon'$$

and $(1 / b_n) \rightarrow 1/b$. Now consider $\lim a_n / b_n = a_n * 1/b_n$. Using part (3) of the Algebraic Limit Theorem, we know $(a_n / b_n) \rightarrow a/b$. $\blacksquare$

From here on out, I'll use the phrase

1. For $x(n) > 0$ $\forall n$: "$x(n)$ can be made **arbitrarily small**" to mean that $\forall \epsilon > 0$, $\exist N \in \N$ s.t. for all $n \geq N$, $x(n) < \epsilon$.
2. For $x(n)$: "$x(n)$ can be made **arbitrarily close** to $l$" to mean that $\forall \epsilon > 0$, $\exist N \in \N$ s.t. for all $n \geq N$, $|x(n) - l| < \epsilon$.
3. For $x(n)$: "Proposition of $x(n)$ is true for **sufficiently large** $n$" to mean that $\exist N \in \N$ s.t. for all $n \geq N$, proposition of $x(n)$ is true.

**The below theorem shows that the limit behaves intuitively under non-strict orderings.**

**Theorem 2.3.4 (Order Limit Theorem)**: Assume $\lim a_n = a$ and $\lim b_n = b$.

1. If $a_n \geq 0$ for all $n \in \N$ then $a \geq 0$.
2. If $a_n \leq b_n$ for all $n \in \N$, then $a \leq b$.
3. If there exists $c \in \R$ for which $c \leq b_n$ for all $n \in \N$, then $c \leq b$. Similarly, if $a_n \leq c$ for all $n \in \N$.

$$ $$

1. *Proof*: We know $\lim a_n = a$, so $|a_n - a|$ can be arbitrarily small. Assume that $a < 0$. As $|a_n - a|$ can be arbitrarily small, choose $\epsilon = \frac{|a|}{2}$. Thus for sufficiently large $n$,

$$ a_n < a + \frac{|a|}{2} = \frac{a}{2} < 0 \Rightarrow\Leftarrow$$

$\blacksquare$

2. *Proof*: We know $\lim a_n = a$ and $\lim b_n = b$, so $|a_n - a|$ and $|b_n - b|$ can be arbitrarily small. We also know that $a_n \leq b_n$. Note that

$$ a_n \leq b_n \Rightarrow c_n = b_n - a_n \geq 0 $$

By the *Algebraic Limit Theorem*, we know that $\lim (b_n - a_n) = b - a$. By part (1) of the *Order Limit Theorem*, we know 

$$b - a \geq 0 \Rightarrow a \leq b$$

$\blacksquare$

3. We know $\lim b_n = b$, so $|b_n - b|$ can be arbitrarily small. We know there exists $c \in \R$ for which $c \leq b_n$ for all $n \in \N$. Note that,

$$ c \leq b_n \Rightarrow b_n - c \geq 0 $$

By the *Algebraic Limit Theorem*, we know

$$ \lim (b_n - c) = b - c $$

(the above fact is clear when you note that a repeating sequence $(c_n = c) \rightarrow c$ because $|c_n - c| = 0 < \epsilon$). By part (1) of the *Order Limit Theorem*:

$$b_n - c \geq 0 \Rightarrow b - c \geq 0 \Rightarrow b \geq c $$

Now examine $(a_n)$. We know $\lim a_n = a$, so $|a_n - a|$ can be arbitrarily small. We know there exists $c \in \R$ for which $c \geq a_n$ for all $n \in \N$. Note that,

$$ c \geq a_n \Rightarrow c - a_n \geq 0 $$

By the *Algebraic Limit Theorem*, we know

$$ \lim (c - a_n) = c - a $$

$$c - a_n \geq 0 \Rightarrow c - a \geq 0 \Rightarrow c \geq a$$

$\blacksquare$

**Notes**:

- Proof (3) is basically an expansion of letting $a_n = c$ or $b_n = c$ and applying (2)

- Though the theorem takes as given the ordering is true for all elements of our sequences, it will also apply if the ordering assumption is satisfied after some $N \in \N$. That's because a sequence's limit is equal to that of a tail subsequence.

**Train Exercises**: 2.3.3, 2.3.5, 2.3.11

**2.3.3 (Squeeze Theorem)**: Show that if $x_n \leq y_n \leq z_n$ for all $n \in \N$, and if $\lim x_n = \lim z_n = l$, then $\lim y_n = l$ as well.

*Proof*: We know that $\lim x_n = \lim z_n = l$. So for $\epsilon > 0$ and sufficiently large $n \in \N$ we have

$$ |x_n - l| < \epsilon \text{ and } |z_n - l| < \epsilon$$

This means that

$$ x_n \in (l - \epsilon, l + \epsilon)$$
$$ \Leftrightarrow l - \epsilon < x_n < l + \epsilon $$

and

$$ z_n \in (l - \epsilon, l + \epsilon)$$
$$ \Leftrightarrow l - \epsilon < z_n < l + \epsilon $$

We also know that $x_n \leq y_n \leq z_n$ $\forall n \in \N$, so for sufficiently large $n \in \N$ we have

$$ l - \epsilon < x_n \leq y_n \leq z_n < l + \epsilon $$
$$ \Leftrightarrow y_n \in (l - \epsilon, l + \epsilon)$$

Thus, $(y_n)$ also converges to $l$.

**2.3.5**: Let $(x_n)$ and $(y_n)$ be given, and define $(z_n)$ to be the "shuffled" sequence $(x_1, y_1, x_2, y_2, \dots)$. Prove that $(z_n)$ is convergent if and only if $(x_n)$ and $(y_n)$ are both convergent with $\lim x_n = \lim y_n = l$

*Proof*: ($\Leftarrow$): We know that $\lim x_n = \lim y_n = l$. This means for $\epsilon > 0$,  $\exist N_x \in \N$ s.t. for $n \geq N_x$:

$$ x_n \in (l - \epsilon, l + \epsilon)$$

Likewise, $\exist N_y \in \N$ s.t. for $n \geq N_y$:

$$ y_n \in (l - \epsilon, l + \epsilon)$$

Let $M = 2 \max (N_x, N_y)$. Then for $m \geq M$ we have:

$$ m \geq 2 \max (N_x, N_y) \geq 2N_x \geq 2N_x - 1$$
$$\Rightarrow \frac{m+1}{2} \geq N_x$$

and

$$ m \geq 2 \max (N_x, N_y) \geq 2N_y$$
$$\Rightarrow \frac{m}{2} \geq N_y$$

Thus,

$$z_m = \left\{\begin{array}{lr}
        x_{\frac{m + 1}{2}}, & \text{if } m \text{ is odd} \\
        y_{\frac{m}{2}}, & \text{if } m \text{ is even}
        \end{array}\right\} \in (l - \epsilon, l + \epsilon)$$

and $(z_m) \rightarrow l$.

($\Rightarrow$): We know that $(z_n)$ is convergent. Let $\lim z_n = l$. This means for $\epsilon > 0$, $\exist N \in \N$ s.t. for $n > N$:

$$ z_n \in (l - \epsilon, l + \epsilon)$$

We also know by definition of $(z_n)$ that:

$$z_n = \left\{\begin{array}{lr}
        x_{\frac{n + 1}{2}}, & \text{if } m \text{ is odd} \\
        y_{\frac{n}{2}}, & \text{if } m \text{ is even}
        \end{array}\right\}$$

Choose $M_x = \frac{N + 1}{2}$. Note that $z_N = x_{M_x}$. Note that for $m > M_x$ and $m = 2n - 1$:

$$ m > M_x \Rightarrow 2n - 1 > 2N - 1 \Rightarrow n > N $$
$$ \Rightarrow x_m = z_{2m - 1} \in (l - \epsilon, l + \epsilon)$$

Thus, $(x_m) \rightarrow l$. Likewise choose $M_y = \frac{N}{2}$. Note that for $m > M_y$ and $m = 2n$:

$$ m > M_y \Rightarrow 2n > 2N \Rightarrow n > N $$
$$ \Rightarrow y_m = z_{2m} \in (l - \epsilon, l + \epsilon)$$

Thus, $\lim x_m = \lim y_m = l$. $\blacksquare$

**2.3.11 (Cesaro Means)** Show that if $(x_n)$ is a convergent sequence, then the sequence given by the averages

$$ y_n = \frac{x_1 + x_2 + x_3 + \cdots x_n}{n} $$

also converges to the limit. Give an example to show that is possible for the sequence $(y_n)$ of averages to converge even if $(x_n)$ does not.

*Proof*: We know that $\lim x_n = l \in \R$. Consider the sequence $(y_n = \frac{x_1 + x_2 + x_3 + \cdots x_n}{n})$. Consider:

$$ |y_n - l| = |\frac{x_1 + x_2 + x_3 + \cdots x_n}{n} - l| = |\frac{x_1 - l}{n} + \frac{x_2 - l}{n} + \cdots + \frac{x_n - l}{n}|$$

Consider $\epsilon > 0$. As $(x_n)$ is a convergent sequence, $\exist N \in \N$ s.t. for $n > N$

$$\tag{1} |x_n - l| < \frac{\epsilon}{2}$$

Now consider $|y_n - l|$ for $n > N$:

$$ |y_n - l| = |\frac{x_1 + x_2 + x_3 + \cdots x_n}{n} - l| = |\frac{x_1 + \cdots + x_N - Nl}{n} + \frac{x_{N+1} +  \cdots + x_n - (n - N)l}{n}|$$

We know $(x_n)$ is a bounded seqeuence because it converges. Let $M \geq |x_n|$ $\forall n \in \N$ be an upper bound for $(x_n)$. This means

$$ |y_n - l| = |\frac{x_1 + \cdots + x_N - Nl}{n} + \frac{x_{N+1} +  \cdots + x_n - (n - N)l}{n}|$$
$$ \leq |\frac{NM - Nl}{n} + \frac{x_{N+1} +  \cdots + x_n - (n - N)l}{n}|$$
$$ \leq |\frac{NM - Nl}{n}| + |\frac{x_{N+1} - l}{n}| + \cdots + |\frac{x_n - l}{n}|$$

Remember that we chose $N$ to be such that for $n > N$ $x_n$ is within $\frac{\epsilon}{2}$ of $l$ (i.e. equation (1)). This further refines our inequality:

$$|y_n - l| \leq |\frac{NM - Nl}{n}| + |\frac{\epsilon}{2n}| + \cdots + |\frac{\epsilon}{2n}|$$

Noting that $\epsilon$ is positive, we can drop the absolute values and clean up:

$$|y_n - l| \leq N\frac{|M - l|}{n} + \frac{(n - N)\epsilon}{2n}$$

Now we can solve for an $n$ s.t. $|y_n - l| < \epsilon$:

$$|y_n - l| \leq N\frac{|M - l|}{n} + \frac{(n - N)\epsilon}{2n} < \epsilon$$
$$\Rightarrow 2N|M - l| + n\epsilon - N\epsilon < 2n\epsilon$$
$$\Rightarrow 2N|M - l| - N\epsilon < 2n\epsilon - n\epsilon$$
$$\Rightarrow 2N|M - l| - N\epsilon < n\epsilon$$
$$\Rightarrow \frac{2N|M - l| - N\epsilon}{\epsilon} < n$$

As $\epsilon > 0$, the term on the left of the inequality is always defined. Thus if we chooose $N' > \frac{2N|M - l| - N\epsilon}{\epsilon}$ then

$$|y_n - l| < \epsilon$$

$\blacksquare$

*Example of Cesaro Mean that converges but undlerying sequence does not*: Consider the sequence $(x_n) = (1, 0, 1, 0, \dots)$. As we can produce no $N \in \N$ s.t. for $n \geq N$

$$ |x_n - l| < \frac{1}{2} $$

for any $l \in \R$. However, the sequence of *Cesaro Means* is $(y_n) = (\frac{\lfloor \frac{n+1}{2} \rfloor}{n})$. Let $\epsilon > 0$. Consider

$$|\frac{\lfloor \frac{n+1}{2} \rfloor}{n} - \frac{1}{2}| $$

And since $\frac{\lfloor \frac{n+1}{2} \rfloor}{n} \geq \frac{1}{2}$:


$$|\frac{\lfloor \frac{n+1}{2} \rfloor}{n} - \frac{1}{2}|  \leq |\frac{\frac{n+1}{2}}{n} - \frac{1}{2}| $$
$$= |\frac{1}{2} + \frac{1}{2n} - \frac{1}{2}| $$
$$= |\frac{1}{2n}| $$
$$= \frac{1}{2n} $$

If we choose $N > \frac{1}{2\epsilon}$, then $|y_n - \frac{1}{2}| < \epsilon$. $\blacksquare$

**Test Exercises**: 2.3.7, 2.3.8, 2.3.12


## 2.4 The Monotone Convergence Theorem and a First Look at the Infinite Series

**Definition 2.4.1**: A sequence $(a_n)$ is *increasing* if $a_n \leq a_{n+1}$ for all $n \in \N$ and *decreasing* if $a_n \geq a_{n+1}$ for all $n \in \N$. A sequence is *monotone* if it is either increasing or decreasing.

**Theorem 2.4.2 (Monotone Convergence Theorem)**: If a sequence is monotone and bounded, then it converges.

*Proof*: Let $(a_n)$ be a monotone and bounded sequence.

1. $(a_n)$ is increasing: By the *Axiom of Completeness* and given that $(a_n)$ is bounded, choose $\sup_{m \in \N} a_m$ as the least upper bound of $(a_n)$. Let $\epsilon > 0$. Consider

$$\tag{1} | a_n -  \sup_{m \in \N} a_m | < \epsilon$$

Note that $\exist n \in \N$ such that (1) is true since if none existed, then $\sup_{m \in \N} a_m - \epsilon < \sup_{m \in \N} a_m$ would be a smaller upper bound. Next, note that given $N \in \N$ such that (1) is true, (1) is also true for $n \geq N$ as $a_n \geq a_N$ ($(a_n)$ is increasing). Thus $\forall \epsilon > 0$ $\exist N \in \N$ such that for $n \geq N$, $| a_n -  \sup_{m \in \N} a_m | < \epsilon$. Therefore, $(a_n)$ converges to $ \sup_{m \in \N} a_m$.

2. $(a_n)$ is decreasing: By the definition of bounded sequence, $a_n \in [-M, M]$ $\forall n \in \N$. Thus, by the *Axiom of Completeness*, $\inf_{m \in \N} a_m$ exists. Consider

$$\tag{2} | a_n -  \inf_{m \in \N} a_m | < \epsilon$$

Note that $\exist n \in \N$ such that (2) is true since if none existed, then $\inf_{m \in \N} a_m + \epsilon > \inf_{m \in \N} a_m$ would be a larger lower bound. Next, note that given $N \in \N$ such that (2) is true, (2) is also true for $n \geq N$ as $a_n \leq a_N$ ($(a_n)$ is decreasing). Thus $\forall \epsilon > 0$ $\exist N \in \N$ such that for $n \geq N$, $| a_n -  \inf_{m \in \N} a_m | < \epsilon$. Therefore, $(a_n)$ converges to $ \inf_{m \in \N} a_m$. $\blacksquare$

**Definition 2.4.3**: Let $(b_n)$ be a sequence. An *infinite series* is a formal expression of the form

$$ \sum_{n=1}^{\infty} b_n = b_1 + b_2 + b_3 + \cdots$$

