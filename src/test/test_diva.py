import datasets.divahisdb as diva
import numpy as np

def test_encoding():
    # color = np.zeros((50,50,3), dtype=np.ubyte)
    # color[:,:,2] = 1
    hex_field = np.zeros((2,2), dtype=np.uint32)
    hex_field[:,:] = diva.Annotations.BACKGROUND
    hex_field[1:, 1:] = diva.Annotations.DECORATION
    hex_field[1:, :1] = diva.Annotations.BODY_TEXT | diva.Annotations.DECORATION

    encodings = diva.to_encoding(hex_field)
    target_encoding = np.array([[ 1,  1],
                                [12,  4]])
    print(hex_field)
    print(encodings)

def test_class_vector():
    y = diva.to_class_vector(diva.Annotations.BODY_TEXT)
    assert all(y == [1,0,0,0])
    y = diva.to_class_vector(diva.Annotations.BACKGROUND)
    assert all(y == [0, 0, 0, 1])
    y = diva.to_class_vector(diva.Annotations.BODY_TEXT | diva.Annotations.DECORATION)
    assert all(y == [1, 1, 0, 0])