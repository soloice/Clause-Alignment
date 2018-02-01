# coding=utf-8
from __future__ import print_function
import gensim
import logging
import codecs
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import argparse
import pickle
import os


parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, default=20, help="Iteration over corpus to train word embeddings.")
parser.add_argument("--word_embedding_size", type=int, default=100, help="Word embedding size.")
parser.add_argument("--negative", type=int, default=5, help="# of negative samples.")
parser.add_argument("--corpus_dir", type=str, default="../data/je-liyanhao/ocr-ocr/1-2/", help="Corpus directory.")
parser.add_argument("--corpus_name_list", type=str, default="je1cut.txt.aligned,je2cut.txt.aligned",
                    help="File names of all corpus files, separated by comma.")
parser.add_argument("--vocab_name", type=str, default="all.vocab",
                    help="File name for vocabulary.")
parser.add_argument("--embedding_name", type=str,
                    default="embedding", help="Embedding pickle file name.")
parser.add_argument("--save_text_format", type=bool, default=False,
                    help="Whether or not to save C text format of word embeddings (typically for debug purpose).")
args = parser.parse_args()



def pretrain_word_embeddings_and_dump_vocabulary(dir,
                                                 corpus_name,
                                                 vocab_name=None,
                                                 append_eos=True,
                                                 embedding_name=None,
                                                 save_text_format=False):
    # Load corpus
    corpus = []
    eos_symbol = "<eos>"
    sentence_frequency = dict()
    # Concatenate all corpura
    for name in corpus_name.split(","):
        with codecs.open(os.path.join(dir, name), "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.split()
                if append_eos:
                    tokens.append(eos_symbol)
                corpus.append(tokens)
                for token in set(tokens):
                    sentence_frequency[token] = sentence_frequency.get(token, 0) + 1

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    model = gensim.models.Word2Vec(corpus, min_count=1, workers=4, iter=args.iter,
                                   sg=1, size=args.word_embedding_size, negative=args.negative)

    with open(os.path.join(dir, embedding_name + ".pkl"), "wb") as f:
        pickle.dump(model.wv, f, True)

    # input_embedding is a KeyedVector
    input_embedding = model.wv
    # np.save(embedding_name, input_embedding.syn0)

    if vocab_name is not None:
        # Save vocab
        with codecs.open(os.path.join(dir, vocab_name), mode="w", encoding="utf-8") as f:
            for i in range(len(input_embedding.vocab)):
                w = input_embedding.index2word[i]
                # Format: <ID>, <token>, <frequency>. Separated by white spaces.
                f.write(str(i) + " " + w + " " + str(input_embedding.vocab[w].count) +
                        " " + str(sentence_frequency[w]) + "\n")

    if save_text_format:
        # Save word embeddings in C text format
        input_embedding.save_word2vec_format(os.path.join(dir, embedding_name + ".txt"), binary=False)


if __name__ == "__main__":
    pretrain_word_embeddings_and_dump_vocabulary(dir=args.corpus_dir,
                                                 corpus_name=args.corpus_name_list,
                                                 vocab_name=args.vocab_name,
                                                 embedding_name=args.embedding_name,
                                                 save_text_format=args.save_text_format)
