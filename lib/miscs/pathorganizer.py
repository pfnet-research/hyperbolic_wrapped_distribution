import pathlib
from pathlib import Path


class PathOrganizer(object):

    def __init__(self, root, name, datadir=(Path.home() / 'data')):
        super(PathOrganizer, self).__init__()
        self.name = name

        self._root = normalize_path(root)
        self._datadir = prepare(datadir)

        self._modelsdir = prepare(self._root / name / 'models')
        self._resultsdir = prepare(self._root / name / 'results')
        self._imagesdir = prepare(self._root / name / 'images')
        self._logsdir = prepare(self._root / name / 'logs')

    def root(self, output='pathlib'):
        return normalize_path(self._root, output=output)

    def namedir(self, output='pathlib'):
        return normalize_path(self._root / self.name, output=output)

    def modelsdir(self, output='pathlib'):
        return normalize_path(self._modelsdir, output=output)

    def resultsdir(self, output='pathlib'):
        return normalize_path(self._resultsdir, output=output)

    def imagesdir(self, output='pathlib'):
        return normalize_path(self._imagesdir, output=output)

    def logsdir(self, output='pathlib'):
        return normalize_path(self._logsdir, output=output)

    def datadir(self, output='pathlib'):
        return normalize_path(self._datadir, output=output)

    def option_path(self, output='pathlib'):
        return normalize_path(self._root / self.name / 'option.json',
                              output=output)

    def history_path(self, output='pathlib'):
        return normalize_path(self._resultsdir / 'history.pkl', output=output)


def prepare(path):
    path = normalize_path(path, output='pathlib')
    if not path.exists():
        path.mkdir(parents=True)
    return path


def normalize_path(path, output='pathlib'):
    if output == 'str':
        if type(path) == str:
            return path
        elif type(path) == pathlib.PosixPath:
            return path.as_posix()
        else:
            raise TypeError(
                ('normalize_path() argument must be a string, '
                 'or a pathlib.PosixPath, not \'{}\'').format(path.__class__))
    elif output == 'pathlib':
        if type(path) == str:
            return Path(path)
        elif type(path) == pathlib.PosixPath:
            return path
        else:
            raise TypeError(
                ('normalize_path() arg 0 must be a string, '
                 'or a pathlib.PosixPath, not \'{}\'').format(path.__class__))
    else:
        raise TypeError(
            'normalize_path() arg 1 must be \'str\' or \'pathlib\', not \'{}\''
            .format(output))
