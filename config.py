import os


PATH = 'few_shot'
DATA_PATH = os.path.join(PATH, 'data')
print('DATA_PATH:', DATA_PATH)

EPSILON = 1e-8

if DATA_PATH is None:
    raise Exception('Configure your data folder location in config.py before continuing!')
