import pickle
from pathlib import Path
from data_downloader import get_split_paths, get_meta_path
import numpy

def unpickle(file: Path) -> dict:
    'unpickles file into dict'
    with file.open('rb') as fo:
        file_data = pickle.load(fo, encoding='latin1')

    return file_data

def combine_data(folder: Path) -> tuple[numpy.ndarray,  list]:
    'given a folder combines data into single numpy array and single label list'
    all_data = []
    all_labels = []

    for train_data in folder.iterdir():

        batch_file = unpickle(train_data)

        data = batch_file['data']
        labels = batch_file['labels']

        all_data.append(data)
        all_labels.extend(labels)

    return numpy.vstack(all_data), all_labels

def load_training() -> tuple[numpy.ndarray, list]:
    'returns training data,labels'
    train, _, _ = get_split_paths()

    return combine_data(train)

def load_testing() -> tuple[numpy.ndarray, list]:
    'returns testing data,labels'
    _, test, _ = get_split_paths()

    return combine_data(test)

def load_validation() -> tuple[numpy.ndarray, list]:
    'returns validation data,labels'
    _, _, validation = get_split_paths()

    return combine_data(validation)


def load_meta() -> list[str]:
    'returns labels names'
    meta = get_meta_path()

    meta_file = unpickle(meta)

    return meta_file['label_names']



def main():
    train_data, train_labels = load_training()
    test_data, test_labels = load_testing()
    valid_data, valid_labels = load_validation()

    print(train_data.shape, len(train_labels))
    print(test_data.shape, len(test_labels))
    print(valid_data.shape, len(valid_labels))

    a = load_meta()

    print(a)

if __name__ == '__main__':
    main()
