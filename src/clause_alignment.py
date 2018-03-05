# coding=utf-8
from __future__ import print_function
from __future__ import division
import pulp
import numpy as np
import pickle
import os
import argparse
import codecs
import scipy.stats


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
parser.add_argument("--version", type=str,
                    default="v1", help="Corpus format version. 'v1' or 'v2'. See L340~341.")
args = parser.parse_args()

ALWAYS_CHOOSE = 2.5



def load_word_embedding(pickle_file_name):
    with open(pickle_file_name, "rb") as f:
        wv = pickle.load(f)
        # wv is a gensim.models.keyedvectors.EuclideanKeyedVectors
        # print(type(wv))

        # wv.vocab is a Python dict
        # print(type(wv.vocab))
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


def weighting_function(avg_idx, length):
    # 该函数单调递增，范围从 0 到 1
    # avg_idx = 1, length -> 0 -> 返回值（权重）趋于 0
    # avg_idx = \inf, length -> \inf -> 返回值（权重）趋于 1
    # 事实上，avg_idx = 100 时词频权重为 0.5，avg_idx > 200 时就几乎为 1 了
    x = (avg_idx-100.0) / 100.0
    sigmoid_x = 1. / (1. + np.exp(-x))
    length_penalty = 1. / (1. + np.exp(-length / 2.0))
    return length_penalty * sigmoid_x


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
    # print(similarities)
    similarity_values = [sim for sim, _1, _2 in similarities]

    top_k = 5
    k = min(len(clause1), len(clause2))
    if k > top_k:
        # 长句对取 top k 的相似性算平均值
        score = sum(similarity_values[-top_k:]) / top_k
        similarity_is = [i for _0, i, _2 in similarities[-top_k:]]
        similarity_js = [j for _0, _1, j in similarities[-top_k:]]
        threshold = 0.95
        # 此处允许得分超过 1：设置为 ALWAYS_CHOOSE，防止被 fertility penalty 干掉。长句对齐如果得分很高，那么这些结果比如
        if score > threshold:
            score = ALWAYS_CHOOSE
    else:
        score = sum(similarity_values[-k:]) / (k + 1e-4)
        similarity_is = [i for _0, i, _2 in similarities[-k:]]
        similarity_js = [j for _0, _1, j in similarities[-k:]]
        # 短句取较短句长做归一化
        # 短句相似性向 0 折扣
        if len(clause1) > len(clause2):
            clause = clause2
        else:
            clause = clause1
        # clause = clause.split()
        average_index = sum([word_vector.vocab[w].index for w in clause]) / (len(clause) + 1e-4)

        # index -> 1 (frequent words), len(clause) -> 0 (short sentence), weight -> 0.0, score -> 0.0
        # index -> \inf (rare words, len(clause) -> 5 (relatively long sentence), weight -> 1, score -> original score
        weight = weighting_function(average_index, len(clause))
        score = score * weight
        # for w in clause:
        #     print("word index:", w, word_vector.vocab[w].index)
        # print("\n")

    # if k >= 1:
    #     # This should always happen
    #     # print(similarity_is)
    #     # print(similarity_js)
    #     if (max(similarity_is) == min(similarity_is)) or (max(similarity_js) == min(similarity_js)):
    #         # 额外处理退化情形（所有 top k 或 k 都对应到了一个词上）
    #         rank_correlation = 0.0
    #     else:
    #         rank_correlation = scipy.stats.spearmanr(similarity_is, similarity_js)[0]
    #
    #     # rank_correlation 的范围是 [-1, 1]
    #     # print("Spearman's rho: ", rank_correlation)
    #
    #     # 如果 rank_correlation = 1，匹配分数不做折扣；
    #     # 如果 rank_correlation = 1，匹配分数折扣 0.9 倍；
    #     # 如果 rank_correlation = -1，匹配分数折扣 0.82 倍；
    #     temperature = 10.0
    #     ordering_weight = np.exp((rank_correlation - 1) / temperature)
    #     score = score * ordering_weight


    # print(" ************************ ")
    # show_sentence(clause1)
    # show_sentence(clause2)
    # print("Similarity = ", similarities)
    # print("Score = ", score)
    return score, len(clause1), len(clause2)


def construct_and_solve_matching(score_matrix, D, verbose, problem_name="pre-matching"):
    if verbose:
        print(score_matrix.shape)
        print(score_matrix)

    l1, l2 = score_matrix.shape
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
    align_problem = pulp.LpProblem(problem_name, pulp.LpMaximize)
    raw_objective = [score_matrix[j, k] * matching_vars[j, k] for j in range(l1) for k in range(l2)]

    # 惩罚多匹配：只支持 D <= 4
    penalty_coefficients = [0.0, 0.4, 0.65, 0.75, 0.85]
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
        return result
    else:
        print("Not converged!", pulp.LpStatus[align_problem.status])
        return "Matching Error!"


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
    effective_lengths1 = np.zeros(l1, dtype=np.float32)
    effective_lengths2 = np.zeros(l2, dtype=np.float32)

    for i in range(l1):
        for j in range(l2):
            scores[i, j], el1, el2 = calculate_score(word_vector, s1_clauses[i], s2_clauses[j])
            # 这里有重复计算，不过无所谓了
            effective_lengths1[i] = el1
            effective_lengths2[j] = el2

    # pre-matching
    matching_result = construct_and_solve_matching(scores, D, verbose, "pre-matching")

    if verbose:
        print("pre-matching result: " + matching_result)

    # refine-matching
    if matching_result != "Matching Error!":
        # introduce position bias
        # matched_pairs 形如 ["0-0", "1-3", "2-2", "3-1"]
        matched_pairs = matching_result.split(", ")
        vars1 = [int(pair[:pair.find("-")]) for pair in matched_pairs]
        vars2 = [int(pair[pair.find("-")+1:]) for pair in matched_pairs]
        range1 = min(vars1), max(vars1)
        range2 = min(vars2), max(vars2)

        # similar to Gaussian kernel
        # extra bonus to diagonal var pairs
        bonus = 0.05
        for i in range(l1):
            for j in range(l2):
                if scores[i, j] == ALWAYS_CHOOSE:
                    continue
                dist_to_diagonal = abs((i - range1[0]) * (range2[1] - range2[0]) - (range1[1] - range1[0]) * (j - range2[0]))
                if verbose:
                    print("dist: ", i, j, dist_to_diagonal)
                # if (i - range1[0]) / (range1[1] - range1[0]) $\approx$ (j - range2[0]) / (range2[1] - range2[0]):
                scores[i, j] += bonus * np.exp(-dist_to_diagonal)
                if scores[i, j] > 1.0:
                    scores[i, j] = 1.0

        # 确认匹配的位置附近的短句给奖励
        threshold = 0.97
        vicinity_range, vicinity_bonus = 3, 0.1
        for i in range(l1):
            for j in range(l2):
                if scores[i, j] == ALWAYS_CHOOSE:
                    continue
                # 相邻位置有一个很确信的元素，本身又包含至少一个小句
                if (i > 0 and scores[i-1, j] > threshold) or (j > 0 and scores[i, j-1] > threshold) \
                        or (i+1 < l1 and scores[i+1, j] > threshold) or (j+1 < l2 and scores[i, j+1] > threshold):
                    if effective_lengths1[i] <= vicinity_range or effective_lengths2[j] <= vicinity_range:
                        scores[i, j] += vicinity_bonus
                        if scores[i, j] > 1.0:
                            scores[i, j] = 1.0

        matching_result = construct_and_solve_matching(scores, D, verbose, "refine-matching")

    if matching_result != "Matching Error!":
        return matching_result, s1c, s2c
    else:
        return matching_result, "", ""


def align_all_corpus(word_vector, corpus1, corpus2, output_file, corpus_format="v1"):
    # corpus_format 允许两种格式：
    #   v1 是 corpus1 和 corpus2 各代表一个版本，每行一句语料，互相平行对齐
    #   v2 是 corpus1 自身就是句对齐的语料，每五行一组，分别是：句子 1 元信息，句子 1， 句子 2 元信息，句子 2，空行。corpus2 为空。
    #  output_file 保存了小句之间对齐的结果
    if corpus_format == "v1":
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
    else:
        meta1, s1, meta2, s2 = "", "", "", ""
        with codecs.open(corpus1, "r", encoding="utf-8") as f:
                with codecs.open(output_file, "w", encoding="utf-8") as f3:
                    for cnt, line in enumerate(f):
                        # print(cnt, line)
                        position = cnt % 5
                        if position == 0:
                            meta1 = line.strip()
                        elif position == 1:
                            s1 = line.strip()
                        elif position == 2:
                            meta2 = line.strip()
                        elif position == 3:
                            s2 = line.strip()
                        else:
                            assert position == 4
                            matching_result, s1_clauses, s2_clauses = align_sentence_pair(word_vector, s1, s2)
                            f3.write(meta1 + meta2 + "\n")
                            f3.write(s1_clauses + "\n")
                            f3.write(s2_clauses + "\n")
                            f3.write(matching_result + "\n\n")

                            cnt += 1
                            if cnt % 500 == 0:
                                print("Processed " + str(cnt) + " lines.\n\n")


def unit_test():
    verbose_option = False

    sent1 = u"我重 又 低头 看书 ， 那 是 本 比 尤伊克 的 《 英国 鸟类 史 》 。 文字 部份 我 一般 不感兴趣 ， 但 有 几页 导言 ， 虽说 我 是 孩子 ， 却 不愿 当作 空页 随手 翻过 。 内中 写 到 了 海鸟 生息 之地 ； 写 到 了 只有 海鸟 栖居 的 “ 孤零零 的 岩石 和 海 岬 ” ； 写 到 了 自 南端 林纳斯 尼斯 ， 或纳斯 ， 至 北角 都 遍布 小岛 的 挪威 海岸 ："
    sent2 = u"我重 又 低头 看 我 的 书 — — 我 看 的 是 比 尤伊克 插图 的 《 英国 禽鸟 史 》 。 一般来说 ， 我 对 这 本书 的 文字 部分 不 大 感兴趣 ， 但是 有 几页 导言 ， 虽说 我 还是 个 孩子 ， 倒 也 不能 当作 空页 一翻 而 过 。 其中 讲 到 海鸟 经常 栖息 的 地方 ， 讲 到 只有 海鸟 居住 的 “ 孤寂 的 岩石 和 海 岬 ” ， 讲 到 挪威 的 海岸 ， 从 最南端 的 林讷 斯内斯 角到 最北 的 北角 ， 星罗棋布 着 无数 岛屿 — —"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=True)
    print("refined result: " + matching_result)


    sent1 = u"那天 ， 出去 散步 是 不 可能 了 。 其实 ， 早上 我们 还 在 光秃秃 的 灌木林 中 溜达 了 一个 小时 ， 但 从 午饭 时起 （ 无客 造访 时 ， 里德 太太 很 早就 用 午饭 ） 便 刮起 了 冬日 凛冽 的 寒风 ， 随后 阴云密布 ， 大雨滂沱 ， 室外 的 活动 也 就 只能 作罢 了 。"
    sent2 = u"那天 ， 再 出去 散步 是 不 可能 了 。 没错 ， 早上 我们 还 在 光秃秃 的 灌木林 中 漫步 了 一个 小时 ， 可是 打 从 吃 午饭 起 （ 只要 没有 客人 ， 里德 太太 总是 很早 吃 午饭 ） ， 就 刮起 了 冬日 凛冽 的 寒风 ， 随之而来 的 是 阴沉 的 乌云 和 透骨 的 冷雨 ， 这一来 ， 自然 也 就 没法 再 到 户外 去 活动 了 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)


    sent1 = u"我 倒 是 求之不得 。 我 向来 不 喜欢 远距离 散步 ， 尤其 在 冷飕飕 的 下午 。 试想 ， 阴冷 的 薄暮 时分 回得家 来 ， 手脚 都 冻僵 了 ， 还要 受到 保姆 贝茵 的 数落 ， 又 自觉 体格 不如 伊 丽莎 、 约翰 和 乔治亚 娜 ， 心里 既 难过 又 惭愧 ， 那 情形 委实 可怕 。"
    sent2 = u"这倒 让 我 高兴 ， 我 一向 不 喜欢 远出 散步 ， 尤其 是 在 寒冷 的 下午 。 我 觉得 ， 在 阴冷 的 黄昏时分 回家 实在 可怕 ， 手指 脚趾 冻僵 了 不 说 ， 还要 挨 保姆 贝茜 的 责骂 ， 弄 得 心里 挺 不 痛快 的 。 再说 ， 自己 觉得 身体 又 比 里德 家 的 伊 丽莎 、 约翰 和 乔治 安娜 都 纤弱 ， 也 感到 低人一等 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)



    sent1 = u"“ 见鬼 ， 上 哪儿 去 了 呀 ？ ” 他 接着 说 。 “ 丽茜 ！ 乔琪 ！ ” （ 喊 着 他 的 姐妹 ） “ 琼 不 在 这儿 呐 ， 告诉 妈妈 她 窜 到 雨 地里 去 了 ， 这个 坏 畜牲 ！ ”"
    sent2 = u"“ 见鬼 ， 她 上 哪儿 去 了 ？ ” 他 接着 说 ： “ 丽茜 ！ 乔琪 ！ （ 他 在 叫 他 的 姐妹 ） 琼 不 在 这儿 。 告诉 妈妈 ， 她 跑 到 外面 雨 地里 去 了 — — 这个 坏东西 ！ ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)


    sent1 = u"里德 太太 对此 则 完全 装聋作哑 ， 她 从 来看 不见 他 打 我 ， 也 从来 听不见 他 骂 我 ， 虽然 他 经常 当着 她 的 面 打 我 骂 我 。"
    sent2 = u"里德 太太 呢 ， 在 这种 事情 上 ， 总是 装聋作哑 ， 她 从 来看 不见 他 打 我 ， 也 从来 听不见 他 骂 我 ， 虽然 他 常常 当着 她 的 面 既 打 我 又 骂 我 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 你 这个 狠毒 的 坏孩子 ！ ” 我 说 ， “ 你 简直 像 个 杀人犯 … … 你 是 个 管 奴隶 的 监工 … … 你 像 那班 罗马 暴君 ！ ” 我 看过 哥 尔德 斯密斯 的 《 罗马 史 》 ， 对尼禄 和 卡利 古拉 一类 人 ， 已经 有 我 自己 的 看法 。 我 曾 在 心里 暗暗 拿 约翰 和 他们 作 过 比较 ， 可是 从 没想到 会 这样 大声 地说 出来 。"
    sent2 = u"“ 你 这 男孩 真是 又 恶毒 又 残酷 ！ ” 我 说 。 “ 你 像 个 杀人犯 — — 你 像 个 虐待 奴隶 的 人 — — 你 像 罗马 的 皇帝 ！ ” 我 看过 哥尔 斯密 的 《 罗马 史 》 ， 对尼禄 和 卡里 古拉 等等 ， 已经 有 我 自己 的 看法 。 我 也 默默地 作 过 比较 ， 却 从 没想到 会 大声 地说 出来 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"事实上 ， 我 确实 有点 失常 ， 或者 像 法国人 常说 的 那样 ， 有点儿 不能自制 了 。"
    sent2 = u"事实上 ， 我 有点儿 失常 ， 或者 像 法国人 所说 的 ， 有点儿 超出 我 自己 的 常规 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 真 不 害臊 ！ 真 不 害臊 ！ ” 使女 嚷嚷 道 ， “ 多 吓人 的 举动 哪 ， 爱 小姐 ， 居然 动手 打 一位 年轻 绅士 ， 打起 你 恩人 的 儿子 ， 你 的 小 主人 来 了 ！ ”"
    sent2 = u"“ 真 不要脸 ！ 真 不要脸 ！ ” 那 使女 说 。 “ 多 吓人 的 举动 ， 爱 小姐 ， 居然 打 起 年轻 的 绅士 ， 打起 你 恩人 的 儿子 来 了 ！ 居然 打 你 的 小 主人 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 请 爱 小姐 坐下 吧 ， ” 他 说 。 在 他 那 勉强 而 生硬 的 点头 和 不耐烦 但 还 合乎 礼节 的 口气 中 ， 似乎 还 表达 了 另 一层 意思 ： “ 见鬼 ， 爱 小姐 来 没来 跟 我 有 什么 关系 ？ 这会儿 我 才 不 愿意 搭理 她 哩 。 ”"
    sent2 = u"“ 让 爱 小姐 坐下 吧 ， ” 他 说 。 那 勉强 的 不 自然 的 点头 和 不耐烦 然而 正式 的 语调 中 ， 似乎 有点 什么 东西 要 进一步 表示 ： “ 见鬼 ， 爱 小姐 在 不 在 这儿 ， 和 我 有 什么 关系 ？ 现在 我 可不 愿 招呼 她 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 是 这样 ， ” 那位 好心 的 太太 接应 说 ， 她 现在 明白 我们 在 说 什么 了 ， “ 是 上帝 指引 我 作出 了 这样 的 选择 ， 为 这 我 每天 都 在 感谢 他 。 爱 小姐 是 我 十分 难得 的 伙伴 ， 她 也 是 阿黛尔 和善 细心 的 老师 。 ”"
    sent2 = u"“ 是 的 ， ” 这位 善良 的 妇人 说 ， 她 现在 知道 我们 在 谈 什么 了 ， “ 上帝 引导 我作 了 这个 选择 ， 我 天天 都 在 感谢 。 爱 小姐 对 我 来说 ， 是 个 非常 可贵 的 伴侣 ， 对 阿黛勒 来说 ， 是 个 既 和蔼 又 细心 的 老师 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    # 这个感觉没办法，新旧算法结果都是 0-3, 1-1, 2-2
    sent1 = u"“ 先生 ？ ” 费尔法克斯 太太 说 。 “ 我 扭伤 了 脚 也 得 感谢 她 哩 。 ”"
    sent2 = u"“ 是 吗 ？ ” 菲尔 费 克斯 太太 说 。 “ 我 这次 扭伤 了 筋 ， 还 得 谢谢 她 呢 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 因为 是 在 假期 ， 我 没有 别的 事要 做 ， 所以 我 坐在 那儿 从 早上 一直 画到 中午 ， 从 中午 一直 画到 晚上 。 仲夏 的 白天 很长 ， 能 让 我 专心致志 地 工作 。 ”"
    sent2 = u"“ 我 没有 别的 事 可 做 ， 因为 那 时候 是 假期 ， 我 就 坐 着 从 早上 画到 中午 ， 又 从 中午 画到 晚上 ， 仲夏 白天 很长 ， 对 我 要 埋头工作 的 心情 是 有利 的 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 都 九点 了 。 你 是 怎么 搞 的 ， 爱 小姐 ， 让 阿黛尔 坐 得 这么久 ？ 快带 她 去 睡觉 。 ”"
    sent2 = u"“ 九点 了 ， 爱 小姐 ， 你 让 阿黛勒 坐 这么久 ， 究竟 是 干什么 ？ 带 她 去 睡觉 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 啊 ！ 我 敢肯定 ！ 你 这人 有点儿 特别 ！ ” 他 说 ， “ 你 的 样子 就 像 个 ‘ 小 修女 ’ 似的 ， 古怪 、 安静 、 严肃 而 又 单纯 。 你 坐在 那儿 ， 两手 放在 身前 ， 眼睛 老是 盯 着 地毯 （ 顺便 说 一句 ， 除了 有时 一个劲儿 盯 着 我 的 脸 ， 比如说 就 像 刚才 那样 ） 。"
    sent2 = u"“ 啊 ！ 我 敢肯定 ！ 你 这人 有点 特别 ， ” 他 说 ， “ 你 的 样子 就 像 个 nonnette 。 你 坐在 那里 ， 两只手 放在 前面 ， 眼睛 老是 盯 着 地毯 （ 顺便 提 一下 ， 除了 尖利 地 盯 着 我 的 脸 ， 譬如说 就 像 刚才 那样 ） ， 你 显得 古怪 、 安静 、 庄严 和 单纯 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 我 相信 ， 先生 ， 我 决不会 把 不拘礼节 错 当成 傲慢无礼 的 。 前 一种 我 反倒 喜欢 ， 而后 一种 ， 没有 哪个 生来 自由 的 人肯 低头 忍受 的 ， 哪怕 是 看 在 薪水 的 分上 。 ”"
    sent2 = u"“ 我 肯定 ， 先生 ， 我 决不会 把 不拘礼节 错 认为 蛮横无理 ； 前者 我 是 相当 喜欢 的 ， 后者 ， 却是 任何 一个 自由民 都 不愿 忍受 的 ， 哪怕 是 拿 了 薪俸 ， 也 不愿 忍受 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"可是 ， 我 的 春天 已经 逝去 了 ， 而 把 一朵 法国 小花 留在 了 我 的 手上 。"
    sent2 = u"不管 怎么样 ， 我 的 春天 已经 过去 了 ， 可是 ， 却 把 那朵 法国 小花 留在 我 手上 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    # 匹配结果多了一个 1-5，很奇怪。怀疑是"有意思"词向量没训好
    sent1 = u"费尔法克斯 太太 在 大厅 里 和 仆人 说话 的 声音 惊醒 了 你 ， 当时 你 多么 奇怪 地 脸露 微笑 ， 而且 在 笑 你 自己 ， 简妮特 ！ 你 的 微笑 意味深长 ， 非常 尖刻 ， 似乎 在 讥笑 你 自己 的 想入非非 。"
    sent2 = u"菲尔 费 克斯 太太 在 大厅 里 跟 用人 说话 的 声音 把 你 惊醒 ； 你 多么 奇怪 地 对 自己 微笑 ， 而且 笑 你 自己 啊 ， 简 ！ 你 的 微笑 ， 很 有意思 ； 笑 得 很 机灵 ， 似乎 在 嘲笑 你 自己 想 得出 神 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"“ 你 不 把 一切 都 告诉 我 ， 你 就 肯定 走 不了 ！ ” 我 说 。 “ 我 不太想 现在 就 说 。 ”"
    sent2 = u"“ 你 不 把 一切 都 告诉 我 ， 你 肯定 就 不能 走 ！ ” 我 说 。 “ 现在 我 倒 不想 说 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    # 太难：第一步空配找错，后来纠正反而把对的纠错了
    sent1 = u"他 愿意 把 它 留给 谁 就 留给 谁 ， 现在 他 把 它 留给 了 你 ， 总之 ， 你 拥有 它 是 完全 正当合理 的 ， 你 可以 问心无愧 地 认为 它 完全 属于 你 。 ”"
    sent2 = u"公正 毕竟 还是 允许 你 保留 它 的 ； 你 可以 问心无愧 地 认为 它 完全 属于 你 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"他 的 野心 是 崇高 的 主 的 精神 那种 野心 ， 它 的 目标 是 要 在 那些 被 拯救 出 尘世 的 人们 的 最 前列 占有 一个 位置 — — 这些 人带 着 无罪 之身 站 在 上帝 的 宝座 跟前 ， 共同 分享 耶稣 最后 的 伟大胜利 ， 他们 都 是 被 召唤 、 被 选中 的 人 ， 都 是 忠贞不渝 的 人 。"
    sent2 = u"他 的 野心 是 崇高 的 主 的 精神 的 那种 野心 ， 它 的 目的 是 要 在 那些 受到 拯救 离开 尘世 的 人们 中间 的 第一排 上 占 一个 位置 — — 他们 毫无 罪过 地站 在 上帝 的 宝座 跟前 ， 共享 着 耶稣 最后 的 伟大胜利 ， 他们 都 是 被 召唤 、 被 选中 的 人 ， 而且 也 都 是 忠诚 的 人 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    sent1 = u"他 也许 依然 严厉 ， 他 也许 依然 苛刻 ， 他 也许 依然 野心勃勃 ， 可是 他 的 严厉 是 武士 大心 的 严厉 ， 正是 大心 保卫 他 护送 的 香客 免受 亚坡伦 的 袭击 。"
    sent2 = u"他 也许 是 严厉 的 ， 他 也许 是 苛刻 的 ， 他 也许 还是 野心勃勃 的 ； 可是 ， 他 的 严厉 是 武士 大心 的 那种 严厉 ， 正是 大心 保卫 着 他 所 护送 的 香客 不 受亚 玻伦 的 袭击 。"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    # 后面的 \mathbb{N} 型匹配很难解
    sent1 = u"他 的 苛刻 是 使徒 的 苛刻 ， 使徒 只是 代表 上帝 才 说 ： “ 若 有人 要 跟 从 我 ， 就 当 舍己 ， 背起 他 的 十字架 来 跟 从 我 。 ”"
    sent2 = u"他 的 苛刻 是 使徒 的 苛刻 ， 使徒 只是 为了 上帝 才 说 ： “ 要 跟着 我 的 人 都 要 拋开 自己 ， 拿 起 他 的 十字架 ， 跟 随着 我 。 ”"
    print("=" * 100)
    matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    print("refined result: " + matching_result)

    # sent1 = u""
    # sent2 = u""
    # print("=" * 100)
    # matching_result, s1_clauses, s2_clauses = align_sentence_pair(vectors, sent1, sent2, verbose=verbose_option)
    # print("refined result: " + matching_result)


if __name__ == "__main__":
    vectors = load_word_embedding(os.path.join(args.data_dir, args.embedding))
    normalize = True
    if normalize:
        # 把所有词向量模长归一化
        norms = np.sqrt(np.sum(vectors.syn0 ** 2, axis=1, keepdims=True))
        vectors.syn0 = vectors.syn0 / norms
    print(vectors.syn0.shape)

    unit_test()

    # align_all_corpus(vectors,
    #                  os.path.join(args.data_dir, args.corpus1),
    #                  os.path.join(args.data_dir, args.corpus2),
    #                  output_file=os.path.join(args.data_dir, args.output_file_name),
    #                  corpus_format=args.version)
