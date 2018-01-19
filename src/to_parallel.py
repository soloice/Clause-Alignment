import codecs
import os
import shutil


def pride_and_prejudice_to_parallel(in_path="../data/pp/prid&prejudious-utf8.txt",
                                    out_path="../data/pp/pride_and_prejudice.txt"):
    en_sentences, zh_sentences = [], []
    en_sentence, zh_sentence = "", ""
    prev_is_chinese = True
    with codecs.open(in_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            s = line.rstrip()
            if len(s) < 1:  #
                continue
            if s.startswith("    "):
                # Chinese
                if prev_is_chinese:
                    zh_sentence += s.strip()
                else:
                    en_sentences.append(en_sentence)
                    zh_sentence = s.strip()
                en_sentence = ""
                prev_is_chinese = True
            else:
                # English
                if prev_is_chinese:
                    zh_sentences.append(zh_sentence)
                    en_sentence = s.strip()
                else:
                    en_sentence += " " + s.strip()
                zh_sentence = ""
                prev_is_chinese = False
        if en_sentence != "":
            en_sentences.append(en_sentence)
        if zh_sentence != "":
            zh_sentences.append(zh_sentence)

    with codecs.open(out_path + ".en", "w", encoding="utf-8") as f:
        for sentence in en_sentences:
            f.write(sentence + "\n")

    with codecs.open(out_path + ".zh", "w", encoding="utf-8") as f:
        for sentence in zh_sentences:
            f.write(sentence + "\n")


def people_daily_to_parallel(en_folder="../data/en",
                             zh_folder="../data/zh",
                             target_folder="../data/news"):
    cnt = 0
    for file_name in os.listdir(en_folder):
        file_en = os.path.join(en_folder, file_name)
        file_zh = os.path.join(zh_folder, file_name)
        if os.path.exists(file_zh):
            print(file_name)
            # copy and remove blank lines
            # with codecs.open(file_en, "r", encoding="utf-8") as f1:
            with open(file_en) as f1:
                with codecs.open(os.path.join(target_folder, str(cnt) + ".en"), "w", encoding="utf-8") as f2:
                    for line in f1.readlines():
                        if len(line.strip()) > 1:
                            f2.write(line)
            with codecs.open(file_zh, "r", encoding="GB18030") as f1:
                with codecs.open(os.path.join(target_folder, str(cnt) + ".zh"), "w", encoding="utf-8") as f2:
                    for line in f1.readlines():
                        if len(line.strip()) > 1:
                            f2.write(line.lstrip("　").replace("\r\n", "\n"))
            cnt += 1

if __name__ == "__main__":
    pride_and_prejudice_to_parallel()
    people_daily_to_parallel()
