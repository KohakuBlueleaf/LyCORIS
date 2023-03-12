# Change Log

## 2023/03/12 Update to 0.1.0
* Add cp-decomposition implementation for convolution layer
  * Both LoRA(LoCon) and LoHa can use this more parameter-efficient decomposition
* Add sparse bias for extracted LoRA
  * Will add to training in the future (Maybe)
* Change weight initialization method in LoHa
  * Use lower std to avoid loss to go high or NaN when using normal lr (like 0.5 in Dadap)