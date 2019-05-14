import argparse
import collections
from functools import partial
import pathlib
import re

import ruamel.yaml
from ruamel.yaml.compat import StringIO

import pygments
import pygments.lexers
import pygments.formatters


def _parse_comment_token(comment_token, key):
    if not comment_token:
        return {
            'key': '--{}'.format(key.replace('_', '-')),
            'type': None,
            'description': ''
        }

    txt = comment_token[2].value.rstrip()[2:]
    m = re.match(
        r'^\<(?P<key>.*?):(?P<type>.*?)\>\s*(?P<description>.*)$', txt)
    if not m:
        return {
            'key': '--{}'.format(key.replace('_', '-')),
            'type': None,
            'description': txt
        }
    else:
        ret = m.groupdict()

        if ret['key'] == '':
            ret['key'] = '--{}'.format(key.replace('_', '-'))
        elif len(ret['key'].split(',')) > 1:
            ret['key'] = ret['key'].split(',')

        if ret['type'] == '':
            ret['type'] = None
        else:
            ret['type'] = eval(ret['type'])
        return ret


class Missing(collections.MutableMapping):

    def __init__(self, store):
        super(Missing, self).__init__()

        self._store = store

    def __repr__(self):
        return self._store.__repr__()

    def __getitem__(self, key):
        return self._store[key]

    def __setitem__(self, key, value):
        self._store[key] = value

    def __delitem__(self, key):
        del self._store[key]

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def __getattr__(self, key):
        try:
            return self.__getattribute__(key)
        except AttributeError:
            ret = self._store[key]
            if isinstance(ret, collections.Mapping):
                return Missing(ret)
            else:
                return ret


class Hyperparams(Missing):

    def __init__(self, recipe_path):
        if isinstance(recipe_path, str):
            recipe_path = pathlib.Path(recipe_path)
        self.recipe_path = recipe_path

        with recipe_path.open('r') as f:
            store = ruamel.yaml.YAML().load(f)
        super(Hyperparams, self).__init__(store)

    def _create_parser(self, program_name=None):
        parser = argparse.ArgumentParser(program_name)
        binder = {}
        for category in self._store.keys():
            for key in self._store[category].keys():
                metadata = _parse_comment_token(
                    self._store[category].ca.items.get(key), key)

                value = self._store[category][key]

                if type(value) == ruamel.yaml.scalarfloat.ScalarFloat:
                    value = float(value)

                if not metadata['type']:
                    metadata['type'] = type(value)

                add_argument = partial(
                    parser.add_argument, help=metadata['description'])

                if isinstance(metadata['key'], list):
                    add_argument = partial(
                        add_argument, metadata['key'][0], metadata['key'][1])
                    parser_key = metadata['key'][1]
                else:
                    add_argument = partial(add_argument, metadata['key'])
                    parser_key = metadata['key']

                if metadata['type'] is bool:
                    add_argument = partial(add_argument, action='store_true')
                else:
                    if isinstance(metadata['type'], list):
                        add_argument = partial(
                            add_argument, choices=metadata['type'])
                    else:
                        add_argument = partial(
                            add_argument, type=metadata['type'])
                    add_argument = partial(add_argument, default=value)

                add_argument()
                if parser_key.startswith('--'):
                    parser_key = parser_key[2:].replace('-', '_')
                else:
                    parser_key = parser_key.replace('-', '_')
                binder[key] = (category, parser_key)
        parser.add_argument('--recipe-path', type=str, default=None,
                            help='alternative recipe file')
        return parser, binder

    def parse_args(self, program_name=None, args=None):
        if not args:
            import sys
            args = sys.argv[1:]

        recipe_path_option = [
            option for option in args if option.startswith('--recipe-path')]

        if len(recipe_path_option) == 1:
            if recipe_path_option[0] == '--recipe-path':
                self = Hyperparams(args[args.index('--recipe-path') + 1])
            else:
                self = Hyperparams(recipe_path_option[0].split('=')[1])
        elif len(recipe_path_option) == 0:
            pass
        else:
            raise AttributeError

        parser, binder = self._create_parser(program_name)
        parsed = parser.parse_args(args)

        for key, target in binder.items():
            category, parser_key = target
            self._store[category][key] = getattr(parsed, parser_key)

        return self

    def dump(self, stream):
        ruamel.yaml.YAML().dump(self._store, stream=stream)

    def summary(self):
        stream = StringIO()
        self.dump(stream=stream)
        return (
            'hyper parameters\n'
            '================\n'
            '{}'.format(pygments.highlight(
                stream.getvalue(), pygments.lexers.YamlLexer(),
                pygments.formatters.TerminalFormatter())
            )
        )


if __name__ == '__main__':
    current_file = pathlib.Path(__file__)
    hpt = Hyperparams(current_file.parents[2] / 'recipes/mlp_default.yml')
    hpt = hpt.parse_args(program_name=current_file.name)
    print(hpt.summary())
