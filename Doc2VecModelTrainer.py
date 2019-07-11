import os
import argparse
import glob
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


logging.basicConfig(format="%(asctime)-15s %(levelname)-8s %(message)s", level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser(description='Doc2Vec Model Trainer',
                                     epilog='By Matan Kolath and Merav Mazouz')
    parser.add_argument('-t', '--text', type=str, help='Path to the text chunks', required=True)
    parser.add_argument('-w', '--write', type=str, help='Path of the output file (model)')
    return parser.parse_args()


def tagged_document_generator(path):
    file_generator = glob.iglob(os.path.join(path, '**', 'chunk*'), recursive=True)
    i = 0
    for f in file_generator:
        try :
            with open(f, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    yield TaggedDocument(words=line.split(), tags=[i])
                    i += 1
        except PermissionError as e:
            logging.exception(e)
            continue


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
    docs = tagged_document_generator(args.text)
    model.build_vocab(docs, progress_per=100000)
    model.save('temp_'+args.write)
    docs = tagged_document_generator(args.text)
    try:
        model.train(docs, total_examples=model.corpus_count, epochs=model.epochs)
    except Exception as e:
        logging.exception(e)
        model.save('exe_'+args.write)
        raise
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    model.save(args.write)


if __name__ == '__main__':
    arg = parse_args()
    main(arg)
