"""Step 3: clean every record once (names + addresses), in 500k-row parts, with 2 worker processes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multiprocessing import Pool
from common import WORK, CLEAN, TRAIN, TEST
import learn_maps, preprocess

if __name__ == '__main__':
    learn_maps.load_maps(f'{WORK}/maps.json')          # before the workers start, so they inherit the maps
    with Pool(2) as pool:
        for name in TRAIN + TEST:
            preprocess.preprocess_file(f'{WORK}/{name}.parquet', f'{CLEAN}/{name}', pool)
            print('cleaned', name)
