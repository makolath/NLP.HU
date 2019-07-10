import os
import argparse
import glob
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


start = time.time()


def parse_args():
    parser = argparse.ArgumentParser(description='Doc2Vec Model Trainer',
                                     epilog='By Matan Kolath and Merav Mazouz')
    parser.add_argument('-t', '--text', type=str, help='Path to the text chunks', required=True)
    parser.add_argument('-w', '--write', type=str, help='Path of the output file (model)')
    return parser.parse_args()


def time_to_str(t):
    h = int(t) // 3600
    m = (int(t) - h * 3600) // 60
    s = t - h * 3600 - m * 60
    return str(h) + ":" + str(m) + ":" + str(s)


def tagged_document_generator(path, prefix=''):
    file_generator = glob.iglob(os.path.join(path, '**', 'chunk*'), recursive=True)
    i = 0
    for f in file_generator:
        if i % 10000 == 0:
            print(prefix + ' line ' + str(i) + ' elapsed: ' + time_to_str(time.time()-start))
        with open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                i += 1
                yield TaggedDocument(words=line, tags=str(i))


def main(args):
    doc2vec_params = {
        'vector_size': 500,
        'epochs': 50,
        'dm': 1,
        'window': 5,
        'min_count': 2,
        'max_vocab_size': None,
        'workers': 4,
        'dm_mean': 1,
    }
    model = Doc2Vec(**doc2vec_params)
    docs = tagged_document_generator(args.text, 'vocab')
    model.build_vocab(docs)
    docs = tagged_document_generator(args.text, 'train')
    model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.save(args.write)


if __name__ == '__main__':
    args = parse_args()
    main(args)
