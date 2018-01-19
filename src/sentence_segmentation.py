# coding=utf-8
from __future__ import unicode_literals, print_function
import en_core_web_sm
import codecs
import os
import jieba


def english_sentence_segmentation(file_name="../data/pp/pride_and_prejudice.txt.en"):
    print("Test: ")
    raw_text = 'Hello, world. Here are two sentences.'
    nlp = en_core_web_sm.load()
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    print(sentences)
    with codecs.open(file_name, "r", encoding="utf-8") as f:
        with codecs.open(file_name + ".seg", "w", encoding="utf-8") as f2:
            for raw_text in f:
                doc = nlp(raw_text.strip())
                sentences = [sent.string.strip() for sent in doc.sents]
                f2.write("///".join(sentences) + "\n")


def chinese_sentence_segmentation(file_name="../data/pp/pride_and_prejudice.txt.zh"):
    sentence_boundaries = ['？', '！', '。']
    tailing_symbols = ['”']
    def chinese_split(line):
        sentences = []
        line = line.strip()
        i, s = 0, 0
        while i < len(line):
            if line[i] in sentence_boundaries:
                if i + 1 < len(line) and line[i+1] in tailing_symbols:
                    i += 1
                sentences.append(line[s:i+1])
                s = i + 1
            i += 1
        if s < len(line): # The last sentence
            sentences.append(line[s:])
        return sentences

    with codecs.open(file_name, "r", encoding="utf-8") as f:
        with codecs.open(file_name + ".seg", "w", encoding="utf-8") as f2:
            for raw_text in f:
                sentences = chinese_split(raw_text)
                f2.write("///".join(sentences) + "\n")


def export_paragraphs(path="../data/pp/", subdir="pp-paragraphs/", separator="///"):
    for file_name in os.listdir(path):
        if file_name.endswith(".seg"):
            with codecs.open(path + file_name, "r", encoding="utf-8") as f:
                lang = "zh" if "zh" in file_name else "en"
                with codecs.open(path + "all-" + lang + ".snt", "w", encoding="utf-8") as f2:
                    for i, line in enumerate(f):
                        sentences = line.strip().split(separator)
                        with codecs.open(path + subdir + str(i) + "_" + lang + ".snt", "w", encoding="utf-8") as f3:
                            if lang == "zh":
                                # word segmentation
                                contents = [" ".join(jieba.cut(sentence)) + "\n" for sentence in sentences]
                            else:
                                contents = [sentence + "\n" for sentence in sentences]
                            f3.writelines(contents)
                        f2.writelines(contents)

if __name__ == "__main__":
    # english_sentence_segmentation()
    # chinese_sentence_segmentation()
    export_paragraphs()
