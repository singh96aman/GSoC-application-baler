import os
import time

import pandas as pd

import modules.helper as helper
import importlib
import warnings

#TODO - Remove Warnings
warnings.filterwarnings("ignore")

def main():
    config, mode, project_name = helper.get_arguments()
    project_path = f"projects/{project_name}/"

    if mode == "new_project":
        helper.create_new_project(project_name)
    elif mode == "preprocessing" or mode == "pp":
        pre_processing(project_name, config.path_before_pre_processing, config.input_path)
    elif mode == "train":
        perform_training(config, project_path)
    elif mode == "compress":
        perform_compression(config, project_path)
    elif mode == "decompress":
        perform_decompression(config.save_as_root, config.model_name, project_path)
    elif mode == "evaluate" or mode == "eval":
        perform_plotting(project_path, config)
    elif mode == "info":
        print_info(project_path)
    elif mode == "analysis":
        analysis(project_name, config.input_path,project_path+"decompressed_output/decompressed.pickle")
    elif mode == "all":
        pre_processing(project_name, config.path_before_pre_processing, config.input_path)
        perform_training(config, project_path)
        perform_compression(config, project_path)
        perform_decompression(config.save_as_root, config.model_name, project_path)
        perform_plotting(project_path, config)
        print_info(project_path)
        analysis(project_name, config.input_path,project_path+"decompressed_output/decompressed.pickle")
    else:
        print(f'"{mode}" is not a know command!')

def pre_processing(project_name, input_path, path_before_pre_processing):
    importlib.import_module(f"projects.{project_name}.{project_name}_preprocessing").pre_processing(input_path, path_before_pre_processing)

def analysis(project_name, data_path_before, data_path_after):
    importlib.import_module(f"projects.{project_name}.{project_name}_analysis").analysis(project_name, data_path_before, data_path_after)

def perform_training(config, project_path):
    print("Training with Model - ",config.model_name)
    train_set, test_set, number_of_columns, normalization_features, cleared_col_names = helper.process(config.input_path,config.test_size)
    train_set_norm = helper.normalize(train_set, config.custom_norm, cleared_col_names)
    test_set_norm = helper.normalize(test_set, config.custom_norm, cleared_col_names)
    try:
        config.latent_space_size = int(number_of_columns//config.compression_ratio)
        config.number_of_columns = number_of_columns
    except AttributeError:
        print(config.latent_space_size,config.number_of_columns)
        assert(number_of_columns==config.number_of_columns)

    device = helper.get_device()

    ModelObject = helper.model_init(config.model_name)
    model = ModelObject(
        device=device, n_features=number_of_columns, z_dim=config.latent_space_size
    )

    output_path = project_path + "training/"
    test_data_tensor, reconstructed_data_tensor = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, output_path, config
    )
    test_data = helper.detach(test_data_tensor)
    if config.model_name in ["VanillaVAE", "CNN_VAE"]:
        #Putting 0 as output of model is list with Reconstructed Data, Mu and Var
        reconstructed_data_tensor = reconstructed_data_tensor[0]

    reconstructed_data = helper.detach(reconstructed_data_tensor)
    
    print("Un-normalzing...")
    start = time.time()
    test_data_renorm = helper.renormalize(
        test_data,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )
    reconstructed_data_renorm = helper.renormalize(
        reconstructed_data,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )
    end = time.time()
    print("Un-normalization took:", f"{(end - start) / 60:.3} minutes")
    
    helper.to_pickle(test_data_renorm, output_path + "before.pickle")
    helper.to_pickle(reconstructed_data_renorm, output_path + "after.pickle")
    normalization_features.to_csv(project_path + "model/cms_normalization_features.csv")
    helper.model_saver(model, project_path + "model/model.pt")
    helper.to_pickle(cleared_col_names,project_path+"compressed_output/column_names.pickle")


def perform_plotting(project_path, config):
    helper.plot(project_path)
    helper.loss_plotter(project_path + "training/loss_data.csv", project_path + "plotting/", config)


def perform_compression(config, project_path):
    print("Compressing...")
    start = time.time()
    if config.model_name in ["VanillaVAE", "CNN_VAE"]:
        compressed_mu, compressed_log_var, data_before, cleared_col_names = helper.compress_VAE(
            model_path=project_path + "model/model.pt",
            config=config,
        )
        compressed_mu = helper.detach(compressed_mu)
        compressed_log_var = helper.detach(compressed_log_var)
    else:
        compressed, data_before, cleared_col_names = helper.compress(
            model_path=project_path + "model/model.pt",
            config=config,
        )
        # Converting back to numpyarray
        compressed = helper.detach(compressed)
    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    if config.model_name in ["VanillaVAE", "CNN_VAE"]:
        helper.to_pickle(compressed_mu, project_path + "compressed_output/compressed_mu.pickle")
        helper.to_pickle(compressed_log_var, project_path + "compressed_output/compressed_log_var.pickle")   
    else:
        helper.to_pickle(compressed, project_path + "compressed_output/compressed.pickle")
    
    helper.to_pickle(
        data_before, project_path + "compressed_output/cleandata_pre_comp.pickle"
    )
    helper.to_pickle(cleared_col_names,project_path+"compressed_output/column_names.pickle")


def perform_decompression(save_as_root, model_name, project_path):
    print("Decompressing...")
    cleared_col_names = helper.from_pickle(project_path+"compressed_output/column_names.pickle")
    start = time.time()
    if model_name in ["VanillaVAE", "CNN_VAE"]:
        decompressed = helper.decompress_VAE(
            model_path=project_path + "model/model.pt",
            input_path_mu=project_path + "compressed_output/compressed_mu.pickle",
            input_path_var=project_path + "compressed_output/compressed_log_var.pickle",
           model_name=model_name,
        )
    else:
        decompressed = helper.decompress(
            model_path=project_path + "model/model.pt",
            input_path=project_path + "compressed_output/compressed.pickle",
            model_name=model_name,
        )
    # Converting back to numpyarray
    decompressed = helper.detach(decompressed)
    normalization_features = pd.read_csv(
        project_path + "model/cms_normalization_features.csv"
    )

    decompressed = helper.renormalize(
        decompressed,
        normalization_features["True min"],
        normalization_features["Feature Range"],
    )

    decompressed = pd.DataFrame(decompressed, columns=cleared_col_names)

    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")

    # False by default
    if save_as_root:
        helper.to_root(
            decompressed, cleared_col_names, project_path + "decompressed_output/decompressed.root"
        )
        helper.to_pickle(
            decompressed, project_path + "decompressed_output/decompressed.pickle"
        )
    else:
        helper.to_pickle(
            decompressed, project_path + "decompressed_output/decompressed.pickle"
        )


def print_info(project_path):
    print(
        "================================== \n Information about your compression \n================================== "
    )

    pre_compression = project_path + "compressed_output/cleandata_pre_comp.pickle"
    compressed = project_path + "compressed_output/compressed.pickle"
    decompressed = project_path + "decompressed_output/decompressed.pickle"

    files = [pre_compression, compressed, decompressed]
    q = []
    for i in range(len(files)):
        q.append(os.stat(files[i]).st_size / (1024 * 1024))

    print(
        f"\nCompressed file is {round(q[1] / q[0], 2) * 100}% the size of the original\n"
    )
    print(f"File size before compression: {round(q[0], 2)} MB")
    print(f"Compressed file size: {round(q[1], 2)} MB")
    print(f"De-compressed file size: {round(q[2], 2),} MB")
    print(f"Compression ratio: {round(q[0] / q[1], 2)}")


if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    main()
