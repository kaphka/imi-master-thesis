import datasets.array
import datasets.divahisdb
import experiment.data
import logging

logging.basicConfig(level=logging.DEBUG)
env = experiment.data.Environment()
diva_processed = env.dataset('DIVA_Chen2017_processed')

dataset = datasets.divahisdb.Processed(diva_processed, load=['tiles', 'y'])
tile_set_path = env.dataset('Chen2017_np_tiles')
datasets.array.combine(dataset,tile_set_path)

split = 'validation'
dataset = datasets.divahisdb.Processed(diva_processed, split=split, load=['tiles', 'y'])
tile_set_path = env.dataset('Chen2017_np_tiles')
datasets.array.combine(dataset,tile_set_path, split=split)

split = datasets.divahisdb.Splits.training.name
dataset = datasets.divahisdb.Processed(diva_processed, split=split, load=['tiles', 'y'])
tile_set_path = env.dataset('Chen2017_np_tiles_balanced')
datasets.array.combine(dataset, tile_set_path, split=split, balance=True)

split = datasets.divahisdb.Splits.validation.name
dataset = datasets.divahisdb.Processed(diva_processed, split=split, load=['tiles', 'y'])
tile_set_path = env.dataset('Chen2017_np_tiles_balanced')
datasets.array.combine(dataset, tile_set_path, split=split, balance=True)