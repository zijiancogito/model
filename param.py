MAX_LEN = 500

LAYER_NUM = 1
D_MODEL = 256
SRC_TOKEN_LEN = 2
TRG_TOEKN_LEN = 1
BATCH_SIZE = 1200
H = 8

LABEL_SMOOTH = 0.1

TRAIN_FILE = '../../data/avg/train.csv'
VAL_FILE = '../../data/avg/vali.csv'
TEST_FILE = '../../data/avg/test.csv'

devices = [0, 1, 2, 3, 4]