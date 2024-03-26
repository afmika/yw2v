import math
import numpy as np

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import re


@dataclass
class Token:
    center: str
    context: List[str]
    negative: List[str]


def tokenize(text_path: str, stopword_path: str) -> List[str]:
    text = open(text_path, mode="r", encoding="utf8").read().lower()
    stop_words = (
        open(stopword_path, mode="r", encoding="utf8").read().replace("\n", " ").split()
    )
    text = re.sub(r"[^\w\d\s]", " ", text)
    words = text.split()
    filtered_words = [
        word for word in words if (word.lower() not in stop_words and len(word) > 2)
    ]
    return filtered_words


def prepare_datas(
    text_path: str, stopword_path: str, window=3, neg_count=3
) -> List[Token]:
    tokens = tokenize(text_path, stopword_path)
    ret = []
    for i in range(len(tokens)):
        # neighbour
        context = []
        for c in range(max(0, i - window), min(len(tokens) - 1, i + window)):
            if c != i:
                context.append(tokens[c])
        center = tokens[i]
        # random words that is not a neighbour
        retries = 20
        negative = []
        while retries > 0:
            word = random.choice(tokens)
            if word not in context and word is not center:
                negative.append(word)
                if len(negative) > neg_count:
                    break
            retries -= 1

        ret.append(Token(tokens[i], context, negative))
    return ret


def vectorize(tokens: List[Token]) -> Tuple[List[Tuple[str, str, float]], any]:
    ret = []
    debug_ret = []
    # note, the position actually encodes the text
    for token in tokens:
        for context in token.context:
            ret.append(1.0)
            debug_ret.append((token.center, context, 1.0))
        for negative in token.negative:
            ret.append(0.0)
            debug_ret.append((token.center, negative, 0.0))
    return debug_ret, np.array([ret]).T


def normalize(items):
    return items / np.sqrt((items**2).sum(axis=1)).reshape(-1, 1)


def normalize_flat_np(items):
    return items / np.sqrt(items**2).sum()


def sigmoid(v):
    return 1 / (1 + np.exp(-v))


def finalize_numpy_mat(
    tokens: List[Tuple[str, str, float]], mat
) -> Dict[str, List[float]]:
    ret = {}
    # 1. order is preserved
    # 2. Same words map to the same vector, hence averaging the sum for common hit
    for i in range(len(tokens)):
        w, _c, _label = tokens[i]
        vec = mat[i]
        if w not in ret:
            ret[w] = (vec, 1)
        else:
            ret[w] = (ret[w][0] + vec, ret[w][1] + 1)

    # TODO: how good is this compared to the harmonic mean? Although, it requires vec[i,j]!=0
    return {k: (agg_vec / total).tolist() for k, (agg_vec, total) in ret.items()}


def train_step(main_emb, context_emb, debug_ret, labels, learning_rate=0.1):
    # this actually computes [sig(dot1) sig(dot2) ... ]
    # sumEachRow(N x M * N x M)
    # result => N == reshape ==> N x 1
    # dot product => can get big => word and context are similar => sigmoid maps the result to -1, 1
    output = sigmoid(np.sum(main_emb * context_emb, axis=1)).reshape(-1, 1)

    # Suppose we have u and v, compute dir=u-v, if we want them to get closer or further apart, reduce the cosine gap in that dir!
    # ex.1 error > 0 and u-v > 0, then move v to get close to u
    # ex.2 error < 0 and u-v > 0, then move v to get further apart from u
    update_dirs = context_emb - main_emb

    # N x 1 - N x 1
    # magnitude getting them closer or further apart
    # the error capture the distance anyways so naturally we move proportionnal to how bad we are
    errors = labels - output

    # core idea, the learning rate is just there to fasten up things or slow down
    # N x M
    deltas = update_dirs * errors * learning_rate

    # main_emb += deltas
    # context_emb -= deltas

    # This suprisingly makes it converge faster
    center = {}
    context = {}
    for i in range(len(debug_ret)):
        w, c, _l = debug_ret[i]
        delta = deltas[i]
        center[w] = delta if w not in center else (delta + center[w])
        context[c] = delta if c not in context else (delta + context[c])

    main_emb += np.array([center[debug_ret[i][0]] for i in range(len(debug_ret))])
    context_emb -= np.array([context[debug_ret[i][1]] for i in range(len(debug_ret))])

    return normalize(main_emb), normalize(context_emb), np.average(errors)


def train(pure_tokens: List[Token], dim_emb, *, steps, learning_rate=0.01):
    debug_ret, labels = vectorize(pure_tokens)

    # word embedding (the result)
    main_emb = normalize(np.random.uniform(-0.5, 0.5, (len(labels), dim_emb)))
    # neighbourhood embedding of a token
    context_emb = normalize(np.random.uniform(-0.5, 0.5, (len(labels), dim_emb)))

    print(f"Embedding dim {dim_emb} | Total steps {steps}")
    for step in range(1, steps + 1):
        main_emb, context_emb, err = train_step(
            main_emb, context_emb, debug_ret, labels, learning_rate
        )
        if step % math.floor(steps / 4) == 0:
            print(f"Step {step}/{steps} | avg error {err}")

    return finalize_numpy_mat(debug_ret, main_emb)
