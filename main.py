from typing import Dict, List
import numpy as np
from word2vec import normalize_flat_np, prepare_datas, train

import json
import re
import time
import sys
import hashlib


def relative_cosine_dist(vec, vectors: Dict[str, List[float]], max_items=10):
    focus = np.array(vec)
    len_f = np.sqrt((focus * focus).sum())
    # u.v = |u||v| cos(u, v) => cos(u, v) = u.v / |u||v|
    ret = []
    for k, v in vectors.items():
        dot = (focus * np.array(v)).sum()
        len_v = np.sqrt((np.array(v) * np.array(v)).sum())
        cosine = dot / (len_v * len_f)
        ret.append((k, cosine))
    return ret


def find_closest_vectors(vec, vectors: Dict[str, List[float]], max_items=10):
    cos_dist = relative_cosine_dist(vec, vectors)
    # the closer the cosine is to 1, the better the similarity is
    return sorted(cos_dist, key=lambda tup: tup[1], reverse=True)[:max_items]


def find_closest(word: str, vectors: Dict[str, List[float]], max_items=10):
    word = word.strip()
    if word not in vectors:
        raise Exception(f"{word} is not part of the knowledge base")
    return find_closest_vectors(vectors[word], vectors, max_items)


def find_furthest(word: str, vectors: Dict[str, List[float]], max_items=10):
    word = word.strip()
    if word not in vectors:
        raise Exception(f"{word} is not part of the knowledge base")
    cos_dist = relative_cosine_dist(vectors[word], vectors)
    # the closer the cosine is to -1, the further the similarity is
    return sorted(cos_dist, key=lambda tup: tup[1], reverse=False)[:max_items]


def eval_expr(expr: str, vectors: Dict[str, List[float]], max_items=10):
    __ = lambda s: np.array(vectors[s])
    to_eval = re.sub(r"([A-Za-z]+)", '__("\\1")', expr)
    # print(f"  ## Interpreting as '{to_eval}' ##")
    return find_closest_vectors(normalize_flat_np(eval(to_eval)), vectors, max_items)


def hash_file(file_path: str):
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(1024)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


# ---------------------------------------
# Token config
window = 10
neg_count = 5

# Model config
dim_emb = 50
steps = 5000
learning_rate = 0.05

input_file = "datas/text.txt" if len(sys.argv[1:]) == 0 else sys.argv[1]
print(f"Loading file {input_file}")

t_start = time.time()
pure_tokens = prepare_datas(input_file, "datas/stop.txt", window, neg_count)
vectors = train(pure_tokens, dim_emb, steps=steps, learning_rate=learning_rate)
t_end = time.time()
print(f"Duration ~{round(t_end - t_start, 2)}s")

with open(
    f"output-{hash_file(input_file)}-win{window}-neg{neg_count}-emb{dim_emb}-steps{steps}-lr{learning_rate}.json",
    "w",
) as f:
    f.write(json.dumps(vectors, sort_keys=True, indent=4))
    f.close()

print("Find the closest word or closest resulting word algebra")
print(
    " Closest: hello, simplistic, cats+animals-humans  (depends on the vocabulary set!)"
)
print(" Furthest: !bird, !many")

top = 20

while True:
    val = input(">> ").lower()
    try:
        ret = []
        if re.search("[+\-/*]", val) is None:
            if val.strip().startswith("!"):
                ret = find_furthest(val.removeprefix("!"), vectors, top)
            else:
                ret = find_closest(val, vectors, top)
        else:
            ret = eval_expr(val, vectors, top)
        print(", ".join([f"{k} {np.round(v, 2)}" for k, v in ret]))
    except Exception as e:
        print(f"Error {e}")
    print()
