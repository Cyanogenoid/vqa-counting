# Counting component for VQA

This is the official implementation of our ICLR 2018 paper [Learning to Count Objects in Natural Images for Visual Question Answering][0] in [PyTorch][1].
The core module is fully contained in `counting.py` in either of the two directories.
If you want to use the counting component, that is the only file that you need.

Check out the README's in the `vqa-v2` directory for VQA v2 and `toy` directory for our toy dataset for more specific information on how to train and evaluate on these datasets.

```
@InProceedings{zhang2018vqacount,
  author    = {Yan Zhang and Jonathon Hare and Adam Pr\"ugel-Bennett},
  title     = {Learning to Count Objects in Natural Images for Visual Question Answering},
  booktitle = {International Conference on Learning Representations},
  year      = {2018},
  eprint    = {1802.05766},
  url       = {https://openreview.net/forum?id=B12Js_yRb},
}
```

[0]: https://openreview.net/forum?id=B12Js_yRb
[1]: https://github.com/pytorch/pytorch
[2]: http://visualqa.org/
