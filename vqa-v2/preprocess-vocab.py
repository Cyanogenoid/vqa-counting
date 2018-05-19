import argparse
import json
from collections import Counter
import itertools

import config
import data
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def main():
    questions = []
    answers = []
    for train in [True, False]:
        questions_path = utils.path_for(train=train, val=not train, question=True)
        answers_path = utils.path_for(train=train, val=not train, answer=True)

        with open(questions_path, 'r') as fd:
            qs = json.load(fd)
        with open(answers_path, 'r') as fd:
            ans = json.load(fd)

        questions += list(data.prepare_questions(qs))
        answers += list(data.prepare_answers(ans))

    question_vocab = extract_vocab(questions, start=1)
    answer_vocab = extract_vocab(answers, top_k=config.max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    with open(config.vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
