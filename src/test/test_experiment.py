import experiment.data as d


def test_load_config():
    env = d.Environment()
    dataset = env.dataset('MNIST')
    assert env.models_folder is not None
    assert env.datasets_path == dataset.parent

def test_train_log():
    log = d.TrainLog()
    print(log.save_directory)
    print(log.log_directory)

    assert log.log_name == 'conf'