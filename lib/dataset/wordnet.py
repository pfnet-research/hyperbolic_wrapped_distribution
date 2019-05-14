import pathlib
from itertools import count
from collections import defaultdict

import numpy as np
from nltk.corpus import wordnet as wn

from chainer import dataset


def generate_dataset(output_dir, with_mammal=False):

    output_path = pathlib.Path(output_dir) / 'noun_closure.tsv'

    # make sure each edge is included only once
    edges = set()
    for synset in wn.all_synsets(pos='n'):
        # write the transitive closure of all hypernyms of a synset to file
        for hyper in synset.closure(lambda s: s.hypernyms()):
            edges.add((synset.name(), hyper.name()))

        # also write transitive closure for all instances of a synset
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                edges.add((instance.name(), hyper.name()))
                for h in hyper.closure(lambda s: s.hypernyms()):
                    edges.add((instance.name(), h.name()))

    with output_path.open('w') as fout:
        for i, j in edges:
            fout.write('{}\t{}\n'.format(i, j))

    if with_mammal:
        import subprocess
        mammaltxt_path = pathlib.Path(output_dir).resolve() / 'mammals.txt'
        mammaltxt = mammaltxt_path.open('w')
        mammal = (pathlib.Path(output_dir) / 'mammal_closure.tsv').open('w')
        commands_first = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-e', r'\smammal.n.01'],
            ['cut', '-f1'],
            ['sed', r's/\(.*\)/\^\1/g']
        ]
        commands_second = [
            ['cat', '{}'.format(output_path)],
            ['grep', '-f', '{}'.format(mammaltxt_path)],
            ['grep', '-v', '-f', '{}'.format(
                pathlib.Path(__file__).resolve().parent / 'mammals_filter.txt'
            )]
        ]
        for writer, commands in zip([mammaltxt, mammal],
                                    [commands_first, commands_second]):
            for i, c in enumerate(commands):
                if i == 0:
                    p = subprocess.Popen(c, stdout=subprocess.PIPE)
                elif i == len(commands) - 1:
                    p = subprocess.Popen(c, stdin=p.stdout, stdout=writer)
                else:
                    p = subprocess.Popen(
                        c, stdin=p.stdout, stdout=subprocess.PIPE)
                # prev_p = p
            p.communicate()
        mammaltxt.close()
        mammal.close()


def parse_seperator(line, length, sep='\t'):
    d = line.strip().split(sep)
    if len(d) == length:
        w = 1
    elif len(d) == length + 1:
        w = int(d[-1])
        d = d[:-1]
    else:
        raise RuntimeError('Malformed input ({})'.format(line.strip()))
    return tuple(d) + (w,)


def parse_tsv(line, length=2):
    return parse_seperator(line, length, '\t')


def iter_line(file_name, parse_function, length=2, comment='#'):
    with open(file_name, 'r') as fin:
        for line in fin:
            if line[0] == comment:
                continue
            tpl = parse_function(line, length=length)
            if tpl is not None:
                yield tpl


def intmap_to_list(d):
    arr = [None for _ in range(len(d))]
    for v, i in d.items():
        arr[i] = v
    assert not any(x is None for x in arr)
    return arr


def slurp(file_name, parse_function=parse_tsv, symmetrize=False):
    ecount = count()
    enames = defaultdict(ecount.__next__)

    subs = []
    for i, j, w in iter_line(file_name, parse_function, length=2):
        if i == j:
            continue
        subs.append((enames[i], enames[j], w))
        if symmetrize:
            subs.append((enames[j], enames[i], w))
    idx = np.array(subs, dtype=np.int)

    # freeze defaultdicts after training data and convert to arrays
    objects = intmap_to_list(dict(enames))
    print('slurp: file_name={}, objects={}, edges={}'.format(
        file_name, len(objects), len(idx)))
    return idx, objects


class WordnetDataset(dataset.DatasetMixin):

    def __init__(self, indices, objects, num_negatives):
        self.indices = indices[:, :2]
        self.objects = objects
        self.num_negatives = num_negatives
        self.num_total_objects = len(self.objects)

    def __len__(self):
        return len(self.indices)

    def get_example(self, i):
        return np.r_[self.indices[i],
                     np.random.randint(self.num_total_objects, size=self.
                                       num_negatives)]


def load_dataset(num_negatives, symmetrize=False, mammal=False,
                 path='~/data/wordnet'):
    if not mammal:
        file_name = pathlib.Path(path).expanduser() / 'noun_closure.tsv'
    else:
        file_name = pathlib.Path(path).expanduser() / 'mammal_closure.tsv'
    if not file_name.exists():
        generate_dataset(pathlib.Path(path), with_mammal=mammal)
    indices, objects = slurp(file_name.as_posix(), symmetrize=symmetrize)
    return WordnetDataset(indices, objects, num_negatives)


def create_adjacency(indices):
    adjacency = defaultdict(set)
    for i in range(len(indices)):
        s, o = indices[i]
        adjacency[s].add(o)
    return adjacency


def calculate_metrics(dataset, loss, k=5,
                      verbose=False, gpu_mode=False, approx=False):
    from sklearn import metrics
    if verbose:
        from tqdm import tqdm

    if gpu_mode:
        import cupy as xp
    else:
        import numpy as xp

    ranks = []
    ap_scores = []

    adjacency = create_adjacency(dataset.indices)

    iterator = tqdm(adjacency.items()) if verbose else adjacency.items()
    batch_size = dataset.num_total_objects // 100
    # batch_size = None
    for i, (source, targets) in enumerate(iterator):
        # if approx and i % 100 != 0:
        if i % 10000 != 0:
            continue
        input_ = xp.c_[
            source * xp.ones(dataset.num_total_objects).astype(xp.int),
            xp.arange(dataset.num_total_objects)]
        _energies = loss.calculate_energy(
            input_, k=k, batch_size=batch_size).flatten()
        if gpu_mode:
            _energies = xp.asnumpy(_energies)
        _energies[source] = 1e+12
        _labels = np.zeros(dataset.num_total_objects)
        _energies_masked = _energies.copy()
        _ranks = []
        for o in targets:
            _energies_masked[o] = np.Inf
            _labels[o] = 1
        ap_scores.append(metrics.average_precision_score(_labels, -_energies))
        for o in targets:
            ene = _energies_masked.copy()
            ene[o] = _energies[o]
            r = np.argsort(ene)
            _ranks.append(np.where(r == o)[0][0] + 1)
        ranks += _ranks
        if verbose and i % max(len(iterator) // 100, 1) == 0:
            tqdm.write('[{}/{}] mAP: {:.4f}, rank: {:.1f}'.format(
                i, len(iterator), np.mean(ap_scores), np.mean(ranks)))
    return np.mean(ranks), np.mean(ap_scores)


if __name__ == '__main__':
    import argparse
    import nltk
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str,
                        default='results',
                        help='output file name')
    parser.add_argument('--mammal', action='store_true')
    parser.add_argument('--symmetrize', action='store_true')
    args = parser.parse_args()
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print('wordnet dataset is not found, start download')
        nltk.download('wordnet')
    print('generate dataset')
    generate_dataset(args.output, with_mammal=args.mammal)
    if not args.mammal:
        file_name = pathlib.Path(args.output) / 'noun_closure.tsv'
    else:
        file_name = pathlib.Path(args.output) / 'mammal_closure.tsv'
    slurp(file_name.as_posix(), symmetrize=args.symmetrize)
