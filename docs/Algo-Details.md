# Algorithm explanation


## Basic Idea
Linear Layer:

$Y = W \cdot X$

finetuning is to get the $W'$

$Y = W \cdot X + W' \cdot X$

and normally $shape(W') = shape(W)$

---

## Conventional Methods
[Ref](https://arxiv.org/abs/2106.09685)

Linear:

$Y_{out \times batch} = W_{out \times in} \cdot X_{in \times batch}$

$\xrightarrow{} Y_{out \times batch} = W_{out \times in} \cdot X_{in \times batch} + Wa_{out \times dim} \cdot Wb_{dim \times in} \cdot X_{in \times batch}$


LoRA for Convolution:
Consider im2col of matmul first:

![image](https://user-images.githubusercontent.com/59680068/221547963-c821b9fa-2825-4b8d-8192-c3109268417f.png)
![image](https://user-images.githubusercontent.com/59680068/221547996-4be14700-1392-4859-9e29-e3e669142a09.png)
$X:[channel, width, height]$

$\xrightarrow{reorder}[c \times kw \times kh, outw \times outh]$

$Kernels: [out, c, kw, kh] \xrightarrow{reshape} [out, c \times kw \times kh]$

$Conv(X, Kernels) = Kernels  \times  X \xrightarrow{reshape} [out, outw, outh]$

and then write down this conventional LoRA for conv layer
$Conv(in, out, ksize, padding, stride)$

$\xrightarrow{}Conv(dim, out, 1)\circ Conv(in, dim, ksize, padding, stride)$


In this method, we can get that
$W' = Wa \cdot Wb$ with $rank(W') \le dim$

---

## Hadamard Product
[Ref](https://arxiv.org/abs/2108.06098)
![image](https://user-images.githubusercontent.com/59680068/223942143-05b5ebff-06c4-4d07-a0eb-037fd6f04e77.png)


consider $W' = Wa \odot Wb$, we can get $rank(W') \le rank(Wa) \times rank(Wb)$.
And then we use conventional method on $Wa$ and $Wb$. Which means it can use 2x dim to get square rank.

**Rank != Information capacity, but they are relative**

based on the experiment result from the paper, it seems like although the rank(Wa) * rank(Wb) is just upper bound, but almost everytime it will produce W' with rank = rank(Wa)*rank(Wb).

### Why custom backward
with $W' = (Wa_1 \cdot Wa_2) \odot (Wb_1 \cdot Wb_2)$, when you need to calc the backpropogation, you will need $\Delta{W'}$ and $Wa$ to calc $\Delta{Wb}$, also $Wb$ for $\Delta{Wa}$.

With pytorch's autograd, this kind of operation will cache the $Wa$ and $Wb$ for calc the backward, which means it will cache 2x size of weight for backward.

To avoid this terrible situation, I impl a custom backward which will reconstruct $Wa$ and $Wb$ when actually needed, this method saved tons of memory.

### CP-Decomposition
[Ref](https://arxiv.org/abs/1412.6553)

As mentioned before, the weight shape for convolution layer is $[out, in, kw, kh]$. And we just unfold it to $[out, in \times kw \times kh]$ for decomposition.

But actually there is a method to decomposition any shape ot tensor called cp decomposition.

Using cp-decomposition in Covolution will be something like:

$\tau: [dim, dim, kw, kh]$ <br>
$x_1: [dim, out]$<br>
$x_2: [dim, in]$<br>
$W' = \tau \times_1 x_1 \times_2 x_2$<br>
$W': [out, in, kw, kh]$

Or write this thing as multiple conv layer:

Conv(in, dim, (1, 1))<br>
↓<br>
Conv(dim, dim, (kw, kh), stride, padding)<br>
↓<br>
conv(dim, out, (1, 1))<br>

For hadamard product implementation, just use 2 different $W'$ and multiply them together.

---

## Kronercker Product

### Definition

If $W_1$ is an a x b matrix and $W_2$ is a c x d matrix, then the Kronecker Product of two matrices is 

$W' = W_1 \otimes W_2$ and an ac x bd matrix.

In meaning of matrix, $W_2$ becomes weight and $W_1$ becomes weight scale of $W_2$

### About rank

And we can decompose $W_2$ using LoRA with rank, r.

$W_2 = Wa_2 \cdot Wb_2$ then $W' = W_1 \otimes (Wa_2 \cdot Wb_2)$

we can get $rank(W') \le rank(W_1) \times rank(Wa_2 \cdot Wb_2)$ and $rank(W_1) \le min(a, b), rank(Wa_2 \cdot Wb_2) \le r$ 

=> $rank(W') \le min(a, b) \times r$

Remember that $min(a, b) \times r$ is upper bound. $rank(W') = min(a, b) \times r$ does not guarantee. Experiment needs.

### Number of parameters

We decompose matrix, $W' = W_1 \otimes (Wa_2 \cdot Wb_2)$.

(# of parameters) = $(a \times b) + (c \times r + r \times d) = a \times b + r \times (c + d)$, m = ac, n = bd.

and suppose best case $a=c= \sqrt{m}, b=d= \sqrt{n}$

then, (# of parameters) = $\sqrt{mn} + r \times (\sqrt{m} + \sqrt{n})$

We can reduce the number of parameters to order of square root maximally.

---

## Sparse Bias
Todo...
