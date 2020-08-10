"""Deliberately uses (colored) prints, instead of proper logging, to avoid
`--logtostderr` (which not only floods the console but also mutes the colors).
"""


class Logger():
    start_str = {
        'red': '\x1b[31m',
        'green': '\x1b[32m',
        'cyan': '\x1b[36m',
        'pink': '\x1b[35m'}
    end_str = '\x1b[0m'

    def __init__(self, loggee=None, debug_mode=False):
        self.debug_mode = debug_mode
        if loggee is None:
            self.prefix = ""
        else:
            self.prefix = "[%s] " % loggee

    def _print(self, txt, color):
        txt = self.prefix + txt
        txt = self.start_str[color] + txt + self.end_str
        print(txt)

    @staticmethod
    def _format(*args):
        return args[0] % tuple(args[1:])

    def warn(self, *args, color='pink'):
        formatted = self._format(*args)
        self._print(formatted, color)

    def warning(self, *args, **kwargs):
        self.warn(*args, **kwargs)

    def error(self, *args, color='red'):
        formatted = self._format(*args)
        self._print(formatted, color)

    def debug(self, *args, color='green'):
        if self.debug_mode:
            formatted = self._format(*args)
            self._print(formatted, color)

    def info(self, *args, color='cyan'):
        formatted = self._format(*args)
        self._print(formatted, color)
