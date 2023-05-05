
class Config:
    max_epoch_num = 100
    max_test_num = 12000
    mini_batch_size = 64
    NO_USE_NORMALIZATION = 0
    is_training = True
    max_patch_num = 140000
    trainImageSize = 128
    ReadColorImage = 1
    isJointTrain = False
    lambda1 = 100
    lambda2 = 10

    step_size = 10  # step size of LR_Schedular
    gamma = 0.1  # decay rate of LR_Schedular


config = Config()