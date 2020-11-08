import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, 'neuro/data')

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_NEG_DIR = os.path.join(TRAIN_DIR, 'neg')
TRAIN_POS_DIR = os.path.join(TRAIN_DIR, 'pos')

TEST_NEG_DIR = os.path.join(TEST_DIR, 'neg')
TEST_POS_DIR = os.path.join(TEST_DIR, 'pos')
