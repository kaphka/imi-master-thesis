import datasets.tiles as t
import datasets.divahisdb as diva
import experiment.data as exp


env = exp.Environment()
target = env.dataset('jigsaw_tiles')
target.mkdir(exist_ok=True)
dataset = diva.HisDBDataset(env.dataset('codices_all'))

t.document_to_tile(dataset, target)

