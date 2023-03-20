
def set_config(c):
    c.input_path          = "data/un_peacekeeping/odp_contributionsbygender.pickle"
    c.path_before_pre_processing = "data/un_peacekeeping/odp_contributionsbygender.csv"
    c.compression_ratio   = 2.0
    c.epochs              = 10
    c.early_stopping      = False
    c.lr_scheduler        = True
    c.patience            = 20
    c.min_delta           = 0
    #c.model_name          = "george_SAE"
    c.model_name          = "VanillaVAE"
    c.custom_norm         = True
    c.l1                  = True
    c.reg_param             = 0.001
    c.RHO                 = 0.05
    c.lr                  = 0.001
    c.batch_size          = 512
    c.save_as_root        = True
    c.test_size           = 0.15
