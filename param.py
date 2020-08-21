MAX_LEN = 500

LAYER_NUM = 6
D_MODEL = 512
SRC_TOKEN_LEN = 2
TRG_TOEKN_LEN =1
BATCH_SIZE = 12000
H = 8

LABEL_SMOOTH = 0.1

TRAIN_FILE = '../../data/avg/train.csv'
VAL_FILE = '../../data/avg/vali.csv'
TEST_FILE = '../../data/avg/test.csv'

devices = [0, 1, 2, 3, 4]