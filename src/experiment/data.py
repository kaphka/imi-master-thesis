from pathlib import Path
import json
from collections import namedtuple


class Environment(object):

    def __init__(self, conf="~/.thesis.conf"):
        conv_path = Path("~/.thesis.conf").expanduser()
        self.config = json.load(conv_path.open())

    @property
    def models_folder(self):
        return Path(self.config['models'])

    @property
    def datasets_path(self):
        return Path(self.config['datasets'])

    @property
    def log_folder(self):
        return self.models_folder / 'logs'

    @property
    def modules(self):
        return Path(self.config['project']) / 'src'

    def dataset(self, name):
        return self.datasets_path / name


class TrainLog(object):
    """
    Save trained models in
    models/model_name/dataset_name/log_name

    Save events in
    models/model_name/dataset_name/log/log_name
    """
    def __init__(self, env=None, dataset_name='data', model=None):
        if env is None:
            env = Environment()
        if model is not None:
            self.model_name = model.name
            self.log_name = model.name + model.conf_str
        else:
            self.model_name = 'model'
            self.log_name = 'conf'

        self.dataset_name = dataset_name

        base = env.models_folder / self.model_name / self.dataset_name
        self.save_directory = base / self.log_name
        self.log_directory  = base / 'log' / self.log_name

