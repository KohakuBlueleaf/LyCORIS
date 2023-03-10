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

based on the experiment result from the paper, it seems like although the rank(Wa) * rank(Wb) is just upper bound, but almost everytime it will produce dW with rank = rank(Wa)*rank(Wb).

### Why custom backward
with $dW = (Wa_1 \cdot Wa_2) \odot (Wb_1 \cdot Wb_2)$, when you need to calc the backpropogation, you will need $\Delta{dW}$ and $Wa$ to calc $\Delta{Wb}$, also $Wb$ for $\Delta{Wa}$.

With pytorch's autograd, this kind of operation will cache the $Wa$ and $Wb$ for calc the backward, which means it will cache 2x size of weight for backward.

To avoid this terrible situation, I impl a custom backward which will reconstruct $Wa$ and $Wb$ when actually needed, this method saved tons of memory.

### Special method for convolution kernels
Todo...

---

## Sparse Bias
Todo...
