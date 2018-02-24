# coding=utf-8
from __future__ import print_function
from __future__ import division
import pulp
import numpy as np
import pickle
import os
import argparse
import codecs



parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="../data/pp/all-trans-pairs/", help="Data folder.")
parser.add_argument("--embedding", type=str,
                    default="embedding.pkl", help="Embedding pickle file name.")
parser.add_argument("--corpus1", type=str,
                    default="src.txt.aligned", help="Name of the first corpus file.")
parser.add_argument("--corpus2", type=str,
                    default="tgt.txt.aligned", help="Name of the second corpus file.")
parser.add_argument("--output_file_name", type=str,
                    default="clause.align.v5", help="Name of output file (clause-level alignment).")
args = parser.parse_args()


def load_word_embedding(pickle_file_name):
    with open(pickle_file_name, "rb") as f:
        wv = pickle.load(f)
        # wv is a gensim.models.keyedvectors.EuclideanKeyedVectors
        print(type(wv))

        # wv.vocab is a Python dict
        print(type(wv.vocab))
        # print(wv.vocab)

        # wv.index2word is a list of words
        # print("/".join(wv.index2word))
        return wv


def chinese_sentence_segmentation(sentence):
    sentence_boundaries = [u'，', u'？', u'！', u'：', u'；', u'。']
    tailing_symbols = [u'”']
    sentences = []
    i, s = 0, 0
    while i < len(sentence):
        if sentence[i] in sentence_boundaries:
            if i + 1 < len(sentence) and sentence[i + 1] in tailing_symbols:
                i += 1
            sentences.append(sentence[s:i+1])
            s = i + 1
        i += 1
    if s < len(sentence): # The last sentence
        sentences.append(sentence[s:])
    return sentences


def show_sentence(sent):
    if type(sent) == unicode:
        # A unicode string, e.g.: "我 爱 你 ， 因为 你 独一无二 。"
        # print(sent)
        return sent
    elif type(sent[0]) == unicode:
        # A list of unicode strings, e.g.: ["我", "爱", "你", "，", "因为", "你", "独一无二", "。"]
        content = " ".join(sent)
        # print(content)
        return content
    else:
        # A list of unicode string lists, e.g.: [["我", "爱", "你", "，"], ["因为", "你", "独一无二", "。"]]
        # chinese_sentence_segmentation(sentence) 的返回类型就是这种
        assert type(sent[0]) == list and type(sent[0][0]) == unicode
        content = "///".join([" ".join(c) for c in sent])
        # print(content)
        return content


def calculate_score(word_vector, clause1, clause2):
    # clause1 和 clause2 都是 unicode string 组成的 list，即形如 ["我", "爱", "你", "，"]
    def filter_punctuations(cl):
        punctuations = [u"。", u"，", u"！", u"？", u"：", u"；", u"“", u"”"]
        res = []
        for w in cl:
            if w not in punctuations:
                res.append(w)
        return res

    # print(show_sentence(clause1))
    # print(show_sentence(clause2))

    # 移除掉小句末尾的标点符号
    clause1 = filter_punctuations(clause1)
    clause2 = filter_punctuations(clause2)

    # print(show_sentence(clause1))
    # print(show_sentence(clause2))

    # 计算相似度的方式 I: 所有词算余弦相似度，然后取 top_k 的平均值

    similarities = []
    identity_match1 = [False] * len(clause1)
    identity_match2 = [False] * len(clause2)
    for i, w1 in enumerate(clause1):
        for j, w2 in enumerate(clause2):
            if w1 == w2:
                # 全同词对只记录第一次配对
                if not identity_match1[i] and not identity_match2[j]:
                    similarities.append((1.0, i, j))
                    identity_match1[i] = True
                    identity_match2[j] = True

                continue
            sim = np.dot(word_vector[w1], word_vector[w2])
            set1, set2 = set(w1), set(w2)
            s = set1 & set2
            ls, ls1, ls2 = len(s), len(set1), len(set2)
            if ls > 0:
                sim = sim + (1-sim) * ls / max(ls1, ls2)

            # print(w1, w2, sim)
            # 之后可以考虑利用 TF-IDF 等进行加权
            similarities.append((sim, i, j))
    similarities.sort()
    print(similarities)
    similarity_values = [sim for sim, _1, _2 in similarities]
    top_k = 5
    k = min(len(clause1), len(clause2))
    if k > top_k:
        score = sum(similarity_values[-top_k:]) / top_k
    else:
        score = sum(similarity_values[-k:]) / (k + 1e-4)

    # print(" ************************ ")
    # show_sentence(clause1)
    # show_sentence(clause2)
    # print("Similarity = ", similarities)
    # print("Score = ", score)
    return score


def align_sentence_pair(word_vector, s1, s2, D=3, verbose=False):
    # D: 最大一对多匹配的度数。目前仅支持小句数量少的一方多匹配小句数量多的一方。
    # s1, s2 都是 unicode 字符串，其中中文已分词，各个词用空格隔开
    # print("s1: ", s1)
    # assert type(s1) == unicode

    s1 = s1.split()
    s2 = s2.split()

    s1_clauses = chinese_sentence_segmentation(s1)
    s2_clauses = chinese_sentence_segmentation(s2)

    s1c = show_sentence(s1_clauses)
    s2c = show_sentence(s2_clauses)


    l1, l2 = len(s1_clauses), len(s2_clauses)

    # scores = np.random.random_integers(-2, 10, [l1, l2])
    scores = np.zeros([l1, l2], dtype=np.float32)
    for i in range(l1):
        for j in range(l2):
            scores[i, j] = calculate_score(word_vector, s1_clauses[i], s2_clauses[j])

    if verbose:
        print(scores)

    # Variables
    matching_vars = pulp.LpVariable.dicts("clause_pairs",
                                          [(j, k) for j in range(l1) for k in range(l2)],
                                          cat='Binary')
    fertility_j = pulp.LpVariable.dicts("fertility_j",
                                        [(d, j) for d in range(2, D + 1) for j in range(l1)],
                                        cat='Binary')

    fertility_k = pulp.LpVariable.dicts("fertility_k",
                                        [(d, k) for d in range(2, D + 1) for k in range(l2)],
                                        cat='Binary')

    # Objective
    align_problem = pulp.LpProblem("matching", pulp.LpMaximize)
    raw_objective = [scores[j, k] * matching_vars[j, k] for j in range(l1) for k in range(l2)]

    # 惩罚多匹配：只支持 D <= 4
    penalty_coefficients = [0.0, 0.0, 0.65, 0.75, 0.85]
    # 惩罚小句少的一方进行多匹配
    penalty_shorter_j, penalty_shorter_k = 0.5 if l1 < l2 else 0.0, 0.5 if l1 > l2 else 0.0
    fertility_penalty_j = [-penalty_coefficients[d] * fertility_j[d, j] + penalty_shorter_j
                           for d in range(2, D + 1) for j in range(l1)]
    fertility_penalty_k = [-penalty_coefficients[d] * fertility_k[d, k] + penalty_shorter_k
                           for d in range(2, D + 1) for k in range(l2)]
    align_problem += pulp.lpSum(raw_objective + fertility_penalty_j + fertility_penalty_k)

    # Constraints
    for j in range(l1):
        align_problem += pulp.lpSum([matching_vars[j, k] for k in range(l2)]) \
                         <= 1 + pulp.lpSum([fertility_j[d, j] for d in range(2, D + 1)])

    for k in range(l2):
        align_problem += pulp.lpSum([matching_vars[j, k] for j in range(l1)]) \
                         <= 1 + pulp.lpSum([fertility_k[d, k] for d in range(2, D + 1)])


    align_problem.solve()
    if pulp.LpStatus[align_problem.status] == "Optimal":
        if verbose:
            for v in matching_vars:
                print(matching_vars[v].name, matching_vars[v].varValue)
            print(pulp.value(align_problem.objective))

        # 提取匹配结果
        result = []
        for j in range(l1):
            for k in range(l2):
                if matching_vars[j, k].varValue > 0.5:  # 即值为 1
                    result.append(str(j) + "-" + str(k))

        result = ", ".join(result)
        return result, s1c, s2c
    else:
        print("Not converged!", pulp.LpStatus[align_problem.status])
        return "Matching Error!", "", ""


def align_all_corpus(word_vector, corpus1, corpus2, output_file):
    # output_file 保存了小句之间对齐的结果
    cnt = 0
    with codecs.open(corpus1, "r", encoding="utf-8") as f1:
        with codecs.open(corpus2, "r", encoding="utf-8") as f2:
            with codecs.open(output_file, "w", encoding="utf-8") as f3:
                for s1, s2 in zip(f1, f2):
                    matching_result, s1_clauses, s2_clauses = align_sentence_pair(word_vector, s1.strip(), s2.strip())
                    f3.write(s1_clauses + "\n")
                    f3.write(s2_clauses + "\n")
                    f3.write(matching_result + "\n\n")

                    cnt += 1
                    if cnt % 100 == 0:
                        print("Processed " + str(cnt) + " lines.\n\n")


if __name__ == "__main__":
    vectors = load_word_embedding(os.path.join(args.data_dir, args.embedding))
    normalize = True
    if normalize:
        # 把所有词向量模长归一化
        norms = np.sqrt(np.sum(vectors.syn0 ** 2, axis=1, keepdims=True))
        vectors.syn0 = vectors.syn0 / norms
    print(vectors.syn0.shape)

    sent1 = u"里德 太太 对此 则 完全 装聋作哑 ， 她 从 来看 不见 他 打 我 ， 也 从来 听不见 他 骂 我 ， 虽然 他 经常 当着 她 的 面 打 我 骂 我 。"
    sent2 = u"里德 太太 呢 ， 在 这种 事情 上 ， 总是 装聋作哑 ， 她 从 来看 不见 他 打 我 ， 也 从来 听不见 他 骂 我 ， 虽然 他 常常 当着 她 的 面 既 打 我 又 骂 我 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=True)
    print(matching_result)



    sent1 = u"“ 你 这个 狠毒 的 坏孩子 ！ ” 我 说 ， “ 你 简直 像 个 杀人犯 … … 你 是 个 管 奴隶 的 监工 … … 你 像 那班 罗马 暴君 ！ ” 我 看过 哥 尔德 斯密斯 的 《 罗马 史 》 ， 对尼禄 和 卡利 古拉 一类 人 ， 已经 有 我 自己 的 看法 。 我 曾 在 心里 暗暗 拿 约翰 和 他们 作 过 比较 ， 可是 从 没想到 会 这样 大声 地说 出来 。"
    sent2 = u"“ 你 这 男孩 真是 又 恶毒 又 残酷 ！ ” 我 说 。 “ 你 像 个 杀人犯 — — 你 像 个 虐待 奴隶 的 人 — — 你 像 罗马 的 皇帝 ！ ” 我 看过 哥尔 斯密 的 《 罗马 史 》 ， 对尼禄 和 卡里 古拉 等等 ， 已经 有 我 自己 的 看法 。 我 也 默默地 作 过 比较 ， 却 从 没想到 会 大声 地说 出来 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=True)
    print(matching_result)

    sent1 = u"事实上 ， 我 确实 有点 失常 ， 或者 像 法国人 常说 的 那样 ， 有点儿 不能自制 了 。"
    sent2 = u"事实上 ， 我 有点儿 失常 ， 或者 像 法国人 所说 的 ， 有点儿 超出 我 自己 的 常规 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=True)
    print(matching_result)

    # align_all_corpus(vectors,
    #                  os.path.join(args.data_dir, args.corpus1),
    #                  os.path.join(args.data_dir, args.corpus2),
    #                  output_file=os.path.join(args.data_dir, args.output_file_name))
