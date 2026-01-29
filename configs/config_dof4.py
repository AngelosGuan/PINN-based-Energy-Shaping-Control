SEED = 60
TEST_SEED = 61232

# adam 
WARM_UP = 20
lr_adam = 5e-4
l2_regu_adam = 1e-7

# lbfgs
lr_lbfgs = 1.0 
max_iter_lbfgs = 100

# batch size
BATCH_SIZE = 16

# hidden layer width
HIDDEN_WIDTH = 512

# neural network depth
NUM_DEPTH = 2

# max grad norm
MAX_GRAD = 100.0

# training size
num_train_data = 10000

# test size
testset_size = 5000

# adaptive sampling
SAMPLE_EVERY = 100 #30
REPLACE_RATE = 0.001 #0.1

EARLY_STAGE_LEN = 100
EARLY_REPLACE = 0.02
SAMPLE_EVERY_EARLY = 5