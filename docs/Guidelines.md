# LyCORIS Guidelines (Tested on SD1)

_written by cybermeow_

The following guidelines are subject to discussion and based on our limited experiments (though with hundreds of trained checkpoints and millions of images generated it is probably by far the largest set of fine-tuning experiments performed so far in the stable diffusion community, we acknowledge it is still rather limited and many aspects have not been investigated).

For further details you can refer to our paper: [Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning to Model Evaluation](https://arxiv.org/abs/2309.14859)

Check also [Resources](Resources.md) to learn more about Stable Diffusion fine-tuning

### Summary table

To be taken with a grain of salt
|                            | Full | LoRA | LoHa | LoKr low factor | LoKr high factor |
| -------------------------- | ---- | ---- | ---- | --------------- | ---------------- |
| Fidelity                   | ★    | ●    | ▲    | ◉               | ▲                |
| Flexibility $^*$           | ★    | ●    | ◉    | ▲               | ● $^†$           |
| Diversity                  | ▲    | ◉    | ★    | ●               | ★                |
| Size $^§$                  | ▲    | ●    | ●    | ●               | ★                |
| Training Speed Linear $^§$ | ★    | ●    | ●    | ★               | ★                |
| Training Speed Conv $^§$   | ●    | ★    | ▲    | ●               | ●                |

★ > ◉ > ● > ▲
[> means better and smaller size is better]

$^*$ Flexibility means anything related to generating images not similar to those in the training set, and combination of multiple concepts, whether they are trained together or not  
$^†$ It may become more difficult to switch base model or combine multiple concepts in this situation  
$^§$ For more details please refer to the tables at the end of the guideline

### Short version

1. If you are looking for the “best” model, then you should stick to native training and curate your dataset. Note you can perform native training with LyCORIS by setting `algo=full` in `network_args`. By so doing, you can use the full model as a LoRA!
2. If space is a more a concern than quality, you probably want to use LoRA. In this case, if you find your model “does not learn well enough” (for example, if some intricate details are not correctly captured), you can try LoKr with low factors (like 4 to 8) and full dimension (by setting dimension to arbitrarily large numbers like 100000).
3. On the contrary, if you find your model “learns too well” (for example, if some unwanted style from training images get leaked or it becomes hard to change pose), your can try LoHa, LoKr with large factors, lower dimension, or even IA3 if you want some fun.
4. Other formats such as DyLoRA and GLoRA are not studied in this experiment, so no recommendations here.


### Extended version

1. Let’s make it clear. Everything depends on the dataset and hyperparameters. The comparaison approximately holds true if LoRA, LoHa, and LoKr have roughly the same size and are otherwise trained with the same hyperparameters. Things may also drastically change if you adjust others things such as `scaled_weight_norm` because the weights produced by different methods do not have the same scale (for example, in our experiments LoRA tend to have much smaller weight than LoHa and LoKr).
2. LoHa’s dampening effect provides new explanation to several observations made in the community so far
    1. LoHa is better at changing character style especially when character is trained with images of a single style: This is because LoHa does not learn the style of the character that much.
    2. LyCORIS is worse at complex characters as for example pointed out by https://rentry.org/59xed3: This refers to especially LoHa and "LoKr with large factors". This occurs because we sacrifice fidelity in exchange for better generalization. Of course you can counter that by adjusting dimension and factor as well.
    3. LoHa is better at learning multiple concepts: Pollution from other concepts are more observed when they are learned “too well”, so this does not happen so much to LoHa thanks to the dampening.
3. Let’s dig into the third point a bit deeper. Is a more underfitted model necessarily better at generalization? This may not be the case as for example illustrated in [SVDiff](https://arxiv.org/abs/2303.11305) (in more detail, they show that combining multiple SVDiffs is worse than combining LoRAs). Moreover, native training seems to be good both at fidelity and its generalization capacity if things are properly tuned, while an ultra small LoKr does not seem to perform well when we switch base model. 
This make me conjecture the following: we need to first fix the model capacity, as roughly measured by the parameter count. Then, there is an inherent trade-off between learning the concept better and better combination of learned concepts, but you can probably improve the two by increasing the model size (so here comes a third tradeoff)!
4. Things become more complex when we consider the effect of captions. Can it be the case that LoHa just attributes the learned concept more uniformly to different words appearing in the caption? This warrants further investigation.
5. Also pay attention to the effect of alpha. Fixing alpha to 1 and half of dimensions is not the same thing when you increase dimension! For the latter you would feel the concept is better learned, while for the former it could be quite the contrary. 
See appendix B.1 of the paper for a relation between alpha and learning rate and initialization.
6. As for the training of convolutional layers or not, the difference is much harder to see as we do not have a systematic way to evaluate "details" or "texture". Recall also that SD is a latent diffusion model, so it is hard to really talk about details without understanding how the VAE reacts change in latent space. I would personally recommend sticking to `preset=attn-mlp` in most cases.
7. Sidenote: I note that some people complain that increasing batch size causes result to be worse. It is important to remember that you should also increase learning rate / training epochs accordingly when you increase batch size to achieve similar results.


### Training time, Vram usage, File size

On RTX 4090, batch 8, 49622 steps, Adamw8bit.  
The time is the estimated time after running for about 500 steps

| Algorithm    | Preset    | Dim [Factor] | Time     | Vram (MiB) | Size                 |
| ------------ | --------- | ------------ | -------- | ---------- | -------------------- |
| LoRA         | attn-mlp  | 1            | 4hr      | 16410      | 1.7 M                |
| LoRA         | attn-mlp  | 8            | 4hr      | 16358      | 9.5 M                |
| LoRA         | attn-only | 32           | 3hr25min | 15222      | 18 M                 |
| LoRA         | attn-mlp  | 32           | 4hr5min  | 15420      | 37 M                 |
| LoRA         | full      | 32           | 4hr30min | 17350      | 75 M                 |
| LoRA         | attn-mlp  | 64           | 4hr5min  | 16168      | 73 M                 |
| LoRA (kohya) |           | 64           | 4hr5min  | 16348      | 73 M                 |
| LoHa         | full      | 4            | 4hr10min | 15960      | 9.6 M                |
| LoHa         | attn-only | 16           | 3hr30min | 15244      | 18 M                 |
| LoHa         | attn-mlp  | 16           | 4hr10min | 16876      | 37 M                 |
| LoHa         | full      | 16           | 5hr40min | 16750      | 75 M                 |
| LoHa         | attn-mlp  | 32           | 4hr10min | 16294      | 73 M                 |
| LoKr         | attn-mlp  | 1 [-1]       | 3hr50min | 16812      | 940 K                |
| LoKr         | attn-mlp  | full [-1]    | 3hr45min | 16806      | 1.6 M                |
| LoKr         | attn-mlp  | 8 [4]        | 3hr55min | 16010      | 2.8 M                |
| LoKr         | attn-mlp  | full [8]     | 3hr40min | 16034      | 12 M                 |
| LoKr         | attn-only | full [4]     | 3hr17min | 15374      | 15 M                 |
| LoKr         | attn-mlp  | full [4]     | 3hr45min | 16116      | 43 M                 |
| LoKr         | full      | full [4]     | 5hr10min | 17916      | 113 M                |
| IA3          |           |              | 3hr7min  | 14954      | 596 K                |
| Full         | attn-only |              | 3hr15min | 15912      | 233 M                |
| Full         | attn-mlp  |              | 3hr50min | 18668      | 673 M                |
| Full         | full      |              | 5h20min  | 23008      | 1.8 G                |
| Full (db)    |           |              | 4hr47min | 22118      | 2 G (with 200mb VAE) |
