# coding=utf-8
from __future__ import print_function
import pulp
import numpy as np


def chinese_sentence_segmentation(sentence):
    sentence_boundaries = [u'，']
    sentences = []
    i, s = 0, 0
    while i < len(sentence):
        if sentence[i] in sentence_boundaries:
            sentences.append(sentence[s:i+1])
            s = i + 1
        i += 1
    if s < len(sentence): # The last sentence
        sentences.append(sentence[s:])
    return sentences


def show_sentence(sent):
    # sent is a list of strings
    if type(sent[0]) == list:
        print("///".join([" ".join(c) for c in sent]))
    else:
        print(" ".join(sent))


s1 = u"凡有 家室 的 男人 ， 总 要 娶 一位 美貌 的 太太 ， 这 已经 成为 了 举世 公认 的 真情 实理 。".split()
s2 = u"饶有家资 的 单身汉 总 要 娶 美女 为 妻 ， 这 是 成为 了 众所周知 的 真理 。".split()

s1_clauses = chinese_sentence_segmentation(s1)
s2_clauses = chinese_sentence_segmentation(s2)

show_sentence(s1)
show_sentence(s2)
show_sentence(s1_clauses)
show_sentence(s2_clauses)


l1 = len(s1_clauses)
l2 = len(s2_clauses)

# Variables
edges = pulp.LpVariable.dicts("clause_pairs",
                              [(j, k) for j in range(l1) for k in range(l2)],
                              cat='Binary')
print(edges)

scores = np.random.random_integers(-2, 10, [l1, l2])
print(scores)


align_problem = pulp.LpProblem("matching", pulp.LpMaximize)

# Objective
align_problem += pulp.lpSum([scores[j, k] * edges[j, k] for j in range(l1) for k in range(l2)])

# Constraints
for j in range(l1):
    align_problem += pulp.lpSum([edges[j, k] for k in range(l2)]) <= 1

for k in range(l2):
    align_problem += pulp.lpSum([edges[j, k] for j in range(l1)]) <= 1


align_problem.solve()
print("Align status: ", pulp.LpStatus[align_problem.status])

for v in edges:
    print(edges[v].name, edges[v].varValue)
print(pulp.value(align_problem.objective))
