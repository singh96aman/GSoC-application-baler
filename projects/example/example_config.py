
def set_config(c):
    c.input_path          = "data/example/example.pickle"
    c.path_before_pre_processing = "data/example/example.root"
    c.compression_ratio   = 2.0
    c.epochs              = 20
    c.early_stopping      = False
    c.lr_scheduler        = True
    c.patience            = 20
    c.min_delta           = 0
    #c.model_name          = "CNN_VAE"
    c.model_name          = "VanillaVAE"
    #c.model_name          = "george_SAE"
    c.custom_norm         = False
    c.l1                  = True
    c.reg_param             = 0.001
    c.RHO                 = 0.05
    c.lr                  = 0.001
    c.lr                  = 0.01 #For VanillaVAE
    c.batch_size          = 512
    c.save_as_root        = True
    c.test_size           = 0.15
