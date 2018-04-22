

class LogInfo():

    def __init__(self, name='info', conv=''):
        self._log_name = name
        self._conf_str = conv

    @property
    def log_name(self):
        return self._log_name + self.conf_str


    @property
    def name(self):
        return self._log_name


    @property
    def conf_str(self):
        return self._conf_str