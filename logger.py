import time

class Logger(object):

    def __init__(self, path):
        self._file = open(path, 'w', encoding=u'utf-8')
        self._file.write(time.strftime("%Y.%m.%d %H:%M:%S", time.localtime(time.time())) + '\n')

    def write(self, key, value):
        assert isinstance(key, str)
        self._file.write(key + '\t' + str(value) + '\n')