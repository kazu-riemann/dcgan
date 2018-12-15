# Standard DCGAN
This is an implementation of GAN [[1]] using [[2]] as a reference.

1. Download CIFAR10 dataset and use the cat class
2. Calculate DCGAN
3. Output real images and generated images every 100 epoch

## Requirements
* Keras==2.2.4
* numpy==1.15.4
* h5py==2.8.0

## Usage
```console
$ python dcgan.py
```

## References
[[1]] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley,
S. Ozair, A. Courville, and Y. Bengio. Generative adversarial nets. In
Advances in Neural Information Processing Systems (NIPS), pages
2672–2680, 2014

[[2]] F. Chollet, Deep learning with Python. Manning Publications, 2018

[1]: http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf
[2]: https://www.manning.com/books/deep-learning-with-python?a_aid=keras&a_bid=76564dff