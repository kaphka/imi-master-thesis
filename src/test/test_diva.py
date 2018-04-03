import datasets.divahisdb as diva
import numpy as np


def test_encoding():
    # color = np.zeros((50,50,3), dtype=np.ubyte)
    # color[:,:,2] = 1
    hex_field = np.zeros((3, 2), dtype=np.uint32)
    hex_field[:, :] = diva.Annotations.BACKGROUND
    hex_field[1:, 1:] = diva.Annotations.DECORATION
    hex_field[1:, :1] = diva.Annotations.BODY_TEXT | diva.Annotations.DECORATION

    encodings = diva.to_encoding(hex_field)
    target_encoding = np.array([[1, 1],
                                [12, 4],
                                [1, 2]])
    print(hex_field)
    print(encodings)


def test_class_vector():
    y = diva.to_class_vector(np.array([diva.Annotations.BODY_TEXT]))
    assert (y == [1, 0, 0, 0]).all()
    y = diva.to_class_vector(np.array([diva.Annotations.BACKGROUND]))
    assert (y == [0, 0, 0, 1]).all()
    y = diva.to_class_vector(np.array([diva.Annotations.BODY_TEXT | diva.Annotations.DECORATION]))
    assert (y == [1, 1, 0, 0]).all()


def test_data_change():
    assert str(diva.change_diva_path('CB55/img/training/file.jpg', set='data')) == \
           'data/img/training/file.jpg'
    assert str(diva.change_diva_path('CB55/img/training/file.jpg', set='data',
                                     split='test', ext='.ext')) == \
           'data/img/test/file.ext'

    assert str(diva.change_diva_path('CB55/img/training/file.jpg', data_format='data')) == \
           'CB55/data/training/file.jpg'