# LoKr, Low-Rank Adaptation with Kronecker Product

Abstract.


---

## 1. Kronecker Product

This section is from [wikipedia](https://en.wikipedia.org/wiki/Kronecker_product). If you familiar with Kronecker Product, you may skip this section.

### 1) Definition

If A is an m × n matrix and B is a p × q matrix, then the Kronecker product A ⊗ B is the pm × qn block matrix:

$\mathbf {A} \otimes \mathbf {B} {\begin{bmatrix}a_{11}\mathbf {B} &\cdots &a_{1n}\mathbf {B} \\\vdots &\ddots &\vdots \\a_{m1}\mathbf {B} &\cdots &a_{mn}\mathbf {B} \end{bmatrix}}$

more explicitly:

$\mathbf{A}\otimes\mathbf{B} = \begin{bmatrix} a_{11} b_{11} & a_{11} b_{12} & \cdots & a_{11} b_{1q} & \cdots & \cdots & a_{1n} b_{11} & a_{1n} b_{12} & \cdots & a_{1n} b_{1q} \\
   a_{11} b_{21} & a_{11} b_{22} & \cdots & a_{11} b_{2q} &
                   \cdots & \cdots & a_{1n} b_{21} & a_{1n} b_{22} & \cdots & a_{1n} b_{2q} \\
   \vdots & \vdots & \ddots & \vdots & & & \vdots & \vdots & \ddots & \vdots \\
   a_{11} b_{p1} & a_{11} b_{p2} & \cdots & a_{11} b_{pq} &
                   \cdots & \cdots & a_{1n} b_{p1} & a_{1n} b_{p2} & \cdots & a_{1n} b_{pq} \\
   \vdots & \vdots & & \vdots & \ddots & & \vdots & \vdots & & \vdots \\
   \vdots & \vdots & & \vdots & & \ddots & \vdots & \vdots & & \vdots \\
   a_{m1} b_{11} & a_{m1} b_{12} & \cdots & a_{m1} b_{1q} &
                   \cdots & \cdots & a_{mn} b_{11} & a_{mn} b_{12} & \cdots & a_{mn} b_{1q} \\
   a_{m1} b_{21} & a_{m1} b_{22} & \cdots & a_{m1} b_{2q} &
                   \cdots & \cdots & a_{mn} b_{21} & a_{mn} b_{22} & \cdots & a_{mn} b_{2q} \\
   \vdots & \vdots & \ddots & \vdots & & & \vdots & \vdots & \ddots & \vdots \\
   a_{m1} b_{p1} & a_{m1} b_{p2} & \cdots & a_{m1} b_{pq} &
                   \cdots & \cdots & a_{mn} b_{p1} & a_{mn} b_{p2} & \cdots & a_{mn} b_{pq}
\end{bmatrix}$


### 2) Properties 

Most of us interest in rank, because rank related with information capacity.

Rank of two matrices after Kronecker Product is 

 $\operatorname{rank} \mathbf{A} \, \operatorname{rank} \mathbf{B} = \operatorname{rank}(\mathbf{A} \otimes \mathbf{B})$ 

But this is in case that A and B has full-rank.

On learning, we don't guarantee this equation holds. So, equation becomes inequation.

 $\operatorname{rank} \mathbf{A} \, \operatorname{rank} \mathbf{B} \leq \operatorname{rank}(\mathbf{A} \otimes \mathbf{B})$ 


---

## 2. LoRA

As you know, Low-Rank Adaptation with rank r for an m x n matrix W is 

$\mathbf{W} = \mathbf{A} \mathbf{B}^T$

$\mathbf{A}$ is an m x r, $\mathbf{B}$ is an n x r matrix, $r << m, n$

---

## 3. LoRA with Kronecker Product

Now we can combine LoRA and Kronecker Product.

If $\mathbf{W}$ is an m x n matrix, there is exist four positive integers a, b, c, d such that m = ac, n = bd.

If $\mathbf{A}$ is an a x r matrix, $\mathbf{B}$ is a b x r matrix, $\mathbf{A}$ is a c x r matrix and  $\mathbf{D}$ is a d x r matrix, then

$\mathbf{W}$ = \($\mathbf{A} \mathbf{B}^T$\) $\otimes$ \($\mathbf{C} \mathbf{D}^T$\)

And $\operatorname{rank} (\mathbf{AB^T}) \, \operatorname{rank} (\mathbf{CD^T}) \leq \operatorname{rank}(\mathbf{AB^T} \otimes \mathbf{CD^T})$ 


All of layers in Stable Diffusion(1.5) model can decompose m, n and a, b, c, d are equal or greater than 16.

Result of factorization is on factorization.txt, (m, n) -> (a, b)⊗(c, d)

---

## 4. Parameter Comparision among various vesion of LoRA

### 1) LoRA, Low-Rank Adaptation. [ref](https://arxiv.org/abs/2106.09685)

If $\mathbf{A}$ is an m x r, $\mathbf{B}$ is an n x r matrix, $r << m, n$, 

$\mathbf{W} = \mathbf{A} \mathbf{B}^T$

(# of parameter) = $r(m + n)$


### 2) LoHa, LoRA with Hadamard Product. [ref](https://arxiv.org/abs/2108.06098)

If $\mathbf{A}$ is an m x r, $\mathbf{B}$ is an n x r matrix, $\mathbf{C}$ is an m x r and $\mathbf{D}$ is an n x r matrix, $r << m, n$, 

$\mathbf{W} = (\mathbf{A} \mathbf{B}^T) \circ (\mathbf{C} \mathbf{D}^T)$

(# of parameter) = $2r(m + n)$

Relative to LoRA, rank of LoHa perform potentially $r^2$. 

So, LoHa can reduce file size by half for maximum rank.

### 3) LoKr, LoRA with Kronecker Product.

If $\mathbf{A}$ is an a x r, $\mathbf{B}$ is a b x r matrix, $\mathbf{C}$ is a c x r and $\mathbf{D}$ is a d x r matrix, $r << m, n$, 

$\mathbf{W} = (\mathbf{A} \mathbf{B}^T) \otimes (\mathbf{C} \mathbf{D}^T)$

(# of parameter) = $r(a+b+c+d) \geq 2r(\sqrt{m} + \sqrt{n})$

Since $m = ac, n = bd$, $a+c \geq 2\sqrt{ac} = 2\sqrt{m}, b+d \geq 2\sqrt{bd} = 2\sqrt{n}$.

It equals when $a=c, b=d$.

Relative to LoRA, rank of LoKr perform potentially $r^2$.

So, LoKr can reduce file size by half of square-root, in experiment ~1/60 times, for maximum rank.

---

## 5. Some results of LoKr

It is on experiment.

rank_lora, optimizer, learning rate, filesize. alpha=rank

16_loRA : lion, unet lr=1.5e-4, TE lr = 7.5e-5, 38,184KB (reference)

4_loRA  : lion, unet lr=1.5e-4, TE lr = 7.5e-5,  9,665KB (-75%)

4_LoHa  : lion, unet lr=1.5e-4, TE lr = 7.5e-5, 19,258KB (-50%)

4_LoKr  : lion, unet lr=3.0e-4, TE lr = 1.5e-4,     633KB (-98%)

8_LoKr  : lion, unet lr=3.0e-4, TE lr = 1.5e-4,   1,027KB (-97%)

16_LoKr  : lion, unet lr=3.0e-4, TE lr = 1.5e-4,   1,817KB (-95%)

images

