from logging import getLogger, StreamHandler, FileHandler, Formatter
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL

LOGFILEPATH_DEFAULT = '/tmp/hyperbolic_wrapped_normal.log'
INFO_FORMAT_DEFAULT = '[%(asctime)s] %(message)s'
DEBUG_FORMAT_DEFAULT = (
    '%(asctime)s - %(name)-12s - %(levelname)-8s - %(message)s')
INFO_DATEFORMAT_DEFAULT = '%Y-%m-%d %H:%M:%S'
DEBUG_DATEFORMAT_DEFAULT = '%Y-%m-%d %H:%M:%S'
levels = (NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL)


class LoggerManager(object):

    def __init__(self, logfilepath=LOGFILEPATH_DEFAULT):
        self.logfilepath = logfilepath
        self.logger = {}

    def __getitem__(self, key):
        return self.logger[key]

    def __setitem__(self, key, value):
        self.logger[key] = value

    def register(self, name, level=INFO, fmt=INFO_FORMAT_DEFAULT,
                 datefmt=INFO_DATEFORMAT_DEFAULT):
        l = getLogger(name)
        l.setLevel(level)

        formatter = Formatter(fmt, datefmt=datefmt)
        sh = StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        fh = FileHandler(self.logfilepath)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        l.addHandler(sh)
        l.addHandler(fh)
        self.logger[name] = l
        return l

    def get(self, name):
        return self.logger[name]

    def setLevel(self, level, name=None):
        assert level in levels, (
            'You have to choose a log level from {}, but you specified: {}'
        ).format(levels, level)

        if name:
            l = self.logger[name]
            l.setLevel(level)
            for h in l.handlers:
                h.setLevel(level)
            return

        for name, l in self.logger.items():
            l.setLevel(level)
            for h in l.handlers:
                h.setLevel(level)

    def setFormatter(self, fmt, datefmt=None):
        if datefmt:
            formatter = Formatter(fmt, datefmt=datefmt)
        else:
            formatter = Formatter(fmt)
        for name, l in self.logger.items():
            for h in l.handlers:
                h.setFormatter(formatter)

    def debugMode(self):
        self.setLevel(DEBUG)
        self.setFormatter(DEBUG_FORMAT_DEFAULT,
                          datefmt=DEBUG_DATEFORMAT_DEFAULT)


logger_manager = LoggerManager(logfilepath=LOGFILEPATH_DEFAULT)
