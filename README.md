# yw2v
Yet another word2vec implementation from scratch.


This is another attempt and simplification of my old [Shiori BOW](https://github.com/afmika/shiori-bow-implementation) which itself is based on [Xin Rong](https://arxiv.org/abs/1411.2738)'s paper.

This one instead uses a [n-gram](https://en.wikipedia.org/wiki/N-gram) based encoding. And is loosely following the following the [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722) paper.

> Note: The only dependency is numpy