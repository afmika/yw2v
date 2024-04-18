from typing import Dict, List
import numpy as np
from word2vec import normalize_flat_np, prepare_datas, train

import json
import re


def find_closest_vectors(vec, vectors: Dict[str, List[float]], max_items=10):
    focus = np.array(vec)
    len_f = np.sqrt((focus * focus).sum())
    # u.v = |u||v| cos(u, v) => cos(u, v) = u.v / |u||v|
    ret = []
    for k, v in vectors.items():
        dot = (focus * np.array(v)).sum()
        len_v = np.sqrt((np.array(v) * np.array(v)).sum())
        cosine = dot / (len_v * len_f)
        ret.append((k, cosine))
    # the closer the cosine is to 1, the better the similarity is
    return sorted(ret, key=lambda tup: tup[1], reverse=True)[:max_items]


def find_closest(word: str, vectors: Dict[str, List[float]], max_items=10):
    word = word.strip()
    if word not in vectors:
        raise Exception(f"{word} is not part of the knowledge base")
    return find_closest_vectors(vectors[word], vectors, max_items)


def eval_expr(expr: str, vectors: Dict[str, List[float]], max_items=10):
    __ = lambda s: np.array(vectors[s])
    to_eval = re.sub(r"([A-Za-z]+)", '__("\\1")', expr)
    # print(f"  ## Interpreting as '{to_eval}' ##")
    return find_closest_vectors(normalize_flat_np(eval(to_eval)), vectors, max_items)


# ---------------------------------------


pure_tokens = prepare_datas("datas/text.txt", "datas/stop.txt", window=4, neg_count=2)
vectors = train(pure_tokens, dim_emb=50, steps=500, learning_rate=0.01)
# print(json.dumps(vectors, sort_keys=True, indent=4))

with open("output.json", "w") as f:
    f.write(json.dumps(vectors, sort_keys=True, indent=4))
    f.close()

print("Find the closest word or closest resulting word algebra")
print(" Examples: hello, dogs, cats+animals-humans  (depends on the vocabulary set!)")

top = 20

while True:
    val = input(">> ").lower()
    try:
        ret = []
        if re.search("[+\-/*]", val) is None:
            ret = find_closest(val, vectors, top)
        else:
            ret = eval_expr(val, vectors, top)
        print(", ".join([f"{k} {np.round(v, 2)}" for k, v in ret]))
    except Exception as e:
        print(f"Error {e}")
    print()
