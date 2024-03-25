from typing import Dict, List
import numpy as np
from word2vec import normalize_flat_np, prepare_datas, train

import json
import re


def find_closest_vector(vec, vectors: Dict[str, List[float]], top=10):
    focus = np.array(vec)
    len_f = np.sqrt((focus * focus).sum())
    # u.v = |u||v| cos(u, v) => cos(u, v) = u.v / |u||v|
    ret = []
    for k, v in vectors.items():
        dot = (focus * np.array(v)).sum()
        len_v = np.sqrt((np.array(v) * np.array(v)).sum())
        cosine = dot / (len_v * len_f)
        ret.append((k, abs(cosine)))
    # the closer the cosine is to 1, the better the similarity is
    return sorted(ret, key=lambda tup: tup[1], reverse=True)[:top]


def find_closest(word: str, vectors: Dict[str, List[float]], top=10):
    word = word.strip()
    if word not in vectors:
        raise Exception(f"{word} is not part of the knowledge base")
    return find_closest_vector(vectors[word], vectors, top)


def eval_expr(expr: str, vectors: Dict[str, List[float]], top=10):
    __ = lambda s: np.array(vectors[s])
    to_eval = re.sub(r"([\w\d]+)", '__("\\1")', expr)
    # print(f"  ## Interpreting as '{to_eval}' ##")
    return find_closest_vector(normalize_flat_np(eval(to_eval)), vectors, top)


# ---------------------------------------


pure_tokens = prepare_datas("datas/text.txt", "datas/stop.txt", window=3, neg_count=6)
vectors = train(pure_tokens, dim_emb=50, steps=100, learning_rate=0.02)
# print(json.dumps(vectors, sort_keys=True, indent=4))


with open("output.json", "w") as f:
    f.write(json.dumps(vectors, sort_keys=True, indent=4))
    f.close()

print("Find the closest word or closest resulting word algebra")
print(
    " Examples: hello, male + female, (animal @ human) * male  (depends on the dataset!)"
)

while True:
    val = input(">> ").lower()
    try:
        if re.search("[+\-/*@]", val) is None:
            print(find_closest(val, vectors))
        else:
            print(eval_expr(val, vectors))
    except Exception as e:
        print(f"Error {e}")
    print()
