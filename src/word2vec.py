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
parser.add_argument("--corpus_dir", type=str, default="../data/pp/all-trans-pairs/", help="Corpus directory.")
parser.add_argument("--embedding_name", type=str, default="embedding", help="Embedding name.")
parser.add_argument("--save_text_format", type=bool, default=False,
                    help="Whether or not to save C text format of word embeddings (typically for debug purpose).")
args = parser.parse_args()


def pretrain_word_embeddings_and_dump_vocabulary(corpus_name,
                                                 vocab_name=None,
                                                 append_eos=True,
                                                 embedding_name=None,
                                                 save_text_format=False):
    # Load corpus
    corpus = []
    eos_symbol = "<eos>"
    with codecs.open(corpus_name, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.split()
            if append_eos:
                tokens.append(eos_symbol)
            corpus.append(line.split() + [eos_symbol])

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Training input and output word embeddings
    #   input_embedding is in syn0, output_embedding is in syn1neg

    model = gensim.models.Word2Vec(corpus, min_count=1, workers=4, iter=args.iter,
                                   sg=1, size=args.word_embedding_size, negative=args.negative)

    with open(os.path.join(embedding_name + ".pkl"), "wb") as f:
        pickle.dump(model.wv, f, True)

    # input_embedding is a KeyedVector
    input_embedding = model.wv
    # np.save(embedding_name, input_embedding.syn0)

    if vocab_name is not None:
        # Save vocab
        with codecs.open(vocab_name, mode="w", encoding="utf-8") as f:
            for i in range(len(input_embedding.vocab)):
                w = input_embedding.index2word[i]
                # Format: <ID>, <token>, <frequency>. Separated by white spaces.
                f.write(str(i) + " " + w + " " + str(input_embedding.vocab[w].count) + "\n")

    if save_text_format:
        # Save word embeddings in C text format
        input_embedding.save_word2vec_format(embedding_name, binary=False)

        # Restore word vectors and print the first 20 dimensions and most similar words for test purpose
        # in_vectors = KeyedVectors.load_word2vec_format(input_embedding_name + ".txt", binary=False)
        # print(in_vectors["the"][:20])
        # print(in_vectors.most_similar("we"))


if __name__ == "__main__":
    # 如果用 all-trans-pairs 训练词向量，那么源词表和目标词表相同
    pretrain_word_embeddings_and_dump_vocabulary(corpus_name=os.path.join(args.corpus_dir, "src.txt.aligned"),
                                                 vocab_name=os.path.join(args.corpus_dir, "all.vocab"),
                                                 embedding_name=os.path.join(args.corpus_dir,
                                                                             args.embedding_name),
                                                 save_text_format=args.save_text_format)
    # pretrain_word_embeddings_and_dump_vocabulary(corpus_name=os.path.join(args.corpus_dir, "tgt.txt.aligned"),
    #                                              vocab_name=os.path.join(args.corpus_dir, "tgt.vocab"),
    #                                              embedding_name=os.path.join(args.corpus_dir,
    #                                                                          args.embedding_name + ".tgt"),
    #                                              save_text_format=args.save_text_format)
