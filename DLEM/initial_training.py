import os
import subprocess
import sys
import time
import pandas as pd
import torch
import shutil
import logging
from csv import writer

sys.path.append('../')

from config.config import get_deep_learning_config, get_classifier_config
from DLEM.train import train, train_ICL, train_ICL_MAML
from utils.deep_learning_util import get_dataset, \
    get_ssl_dataset, \
    get_data_loader, \
    get_model, \
    get_ssl_optimizer_scheduler, \
    get_icl_optimizer_scheduler, \
    get_optimizer_scheduler, \
    train_test_valid_split, \
    serialization, \
    serialization_ssl, \
    create_batches
from pools.label_pool import LabelPool
from pools.ssl_pool import SSLPool
from meta_data.metadata import MetaData
from utils.model_deploy_util import ModelDeployFlag

logger = logging.getLogger(__name__)


class EMStreamTrain:
    def __init__(self):
        self.y_valid = None
        self.y_test = None
        self.y_train = None
        self.X_valid = None
        self.X_test = None
        self.X_train = None
        self.X_icl_new = None
        self.X_icl_old = None
        self.X_ssl = None
        self.dl_config = get_deep_learning_config()
        self.ssl_pool = SSLPool()
        self.label_pool = LabelPool()
        self.m_data = MetaData()
        self.model_flag = ModelDeployFlag()
        self.classifier_config = get_classifier_config()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metadata = self.m_data.get_meta_data()
        self.pre_trained_model = self.metadata["PRE_TRAINED_MODEL_PATH"]

    def execute_model_training(self, dl_error_csv_obj, train_error_csv_obj, i):
        """
        First, entities for pre-training (Incremental Contrastive Learning) are retrieved.
        Second, entities from label-pool secondary pool are read and augmentation is applied if required.
        Third, train, validation and test split are generated.
        Last, DL-model is trained and evaluated.

        :param dl_error_csv_obj: A csv-file to store DL-errors per epoch.
        :param train_error_csv_obj: A csv-file to store DL-train errors.
        :param i: index
        :return: None
        """
        logger.info("Starting DL-based model training.")
        torch.cuda.empty_cache()
        model_flag = self.model_flag.get_flag()
        
        if model_flag == 1 or model_flag == 2:
            entities_icl = self.ssl_pool.retrieve_data(1)
            entities_icl_old = entities_icl[0]
            entities_icl_old = serialization_ssl(entities_icl_old)
            df_icl_old = pd.DataFrame.from_records(entities_icl_old)
            self.X_icl_old = df_icl_old.to_numpy()

            entities_icl_new = entities_icl[1]
            entities_icl_new = serialization_ssl(entities_icl_new)
            df_icl_new = pd.DataFrame.from_records(entities_icl_new)
            self.X_icl_new = df_icl_new.to_numpy()
        elif model_flag == 0:
            entities_ssl = self.ssl_pool.retrieve_data(0)
            entity_pairs_ssl = serialization_ssl(entities_ssl)
            df_ssl = pd.DataFrame.from_records(entity_pairs_ssl)
            self.X_ssl = df_ssl.to_numpy() 
        
        entities = self.label_pool.retrieve_data(self.dl_config["data_aug"])
        entity_pairs = serialization(entities)
        
        df = pd.DataFrame.from_records(entity_pairs)  # Converts entity_pool-points into pandas dataframe.
        self.X_train, self.X_test, self.X_valid, self.y_train, self.y_test, self.y_valid = \
            train_test_valid_split(df)
        
        self.em_stream_train(dl_error_csv_obj, train_error_csv_obj, i)
    
        logger.info("Completed DL-based model training.")

    def em_stream_train(self, dl_error_csv_obj, train_error_csv_obj, i):
        """
        The function creates datasets and entity_pool loaders for pre-training, fine-tuning, test, and valid entity_pool.
        Assigns the model, optimizer, and schedulers if there exists a saved model otherwise
        new model, optimizer, and scheduler is created.
        :return: None
        """
        #############################################
        # Intiliaze data set, data loader and model #
        #############################################
        model_flag = self.model_flag.get_flag()
        
        if model_flag == 0:
            logger.info("Creating contrastive learning dataloader.")
            start_time = time.time()
            trainset_nolabel = get_ssl_dataset(self.X_ssl)
            if self.dl_config["clustering"]:
                trainset_nolabel_loader = create_batches(trainset_nolabel, self.dl_config["BATCH_SIZE_SSL"], self.dl_config["SSL_EPOCHS"])
            else:
                padder_contraLearning = trainset_nolabel.pad
                trainset_nolabel_loader = get_data_loader(1, trainset_nolabel, padder_contraLearning)
            end_time = time.time()
            logger.info(f"Time taken to create CL-loader: {end_time - start_time}")
        elif model_flag == 1 or model_flag == 2:
            logger.info("Creating ICL train dataloaders.")
            start_time = time.time()
            train_icl_old_dataset = get_ssl_dataset(self.X_icl_old)
            train_icl_new_dataset = get_ssl_dataset(self.X_icl_new)
            if self.dl_config["clustering"]:
                icl_old_loader = create_batches(train_icl_old_dataset, self.dl_config["BATCH_SIZE_SSL"], self.dl_config["SSL_EPOCHS"])
                icl_new_loader = create_batches(train_icl_new_dataset, self.dl_config["BATCH_SIZE_SSL"], self.dl_config["SSL_EPOCHS"])
            else:
                padder_old = train_icl_old_dataset.pad
                icl_old_loader = get_data_loader(1, train_icl_old_dataset, padder_old)
                padder_new = train_icl_new_dataset.pad
                icl_new_loader = get_data_loader(1, train_icl_new_dataset, padder_new)

            end_time = time.time()
            logger.info(f"Time taken to create ICL-loader: {end_time - start_time}")

        logger.info("Creating train dataloader.")
        start_time = time.time()
        train_dataset = get_dataset(self.X_train, self.y_train)
        padder = train_dataset.pad
        train_loader = get_data_loader(0, train_dataset, padder, "train")
        end_time = time.time()
        logger.info(f"Time taken to create entity_pool-loader: {end_time - start_time}")

        logger.info("Creating test dataloader.")
        test_dataset = get_dataset(self.X_test, self.y_test)
        test_loader = get_data_loader(0, test_dataset, padder)

        logger.info("Creating validation dataloader.")
        valid_dataset = get_dataset(self.X_valid, self.y_valid)
        valid_loader = get_data_loader(0, valid_dataset, padder)

        lang_model = self.dl_config["lm"]
        logger.info(f"Creating deep learning-based {lang_model}.")
        model = get_model(self.device)
        
        logger.info(f"Creating optimizer and scheduler.")
        if model_flag == 0:
            optimizer, scheduler = get_ssl_optimizer_scheduler(trainset_nolabel, model)
            optimizer_f, scheduler_f = get_optimizer_scheduler(train_dataset, model)       
        elif model_flag == 1 or model_flag == 2:
            if self.dl_config["MAML"]:
                optimizer_new, scheduler_new = get_icl_optimizer_scheduler([], train_icl_new_dataset, model)
                optimizer_f, scheduler_f = get_optimizer_scheduler(train_dataset, model)
            else:
                optimizer_old, scheduler_old, optimizer_new, scheduler_new = get_icl_optimizer_scheduler(train_icl_old_dataset, train_icl_new_dataset, model)
                optimizer_f, scheduler_f = get_optimizer_scheduler(train_dataset, model)

        ##################
        # Model training #
        ##################
        if model_flag == 1 or model_flag == 2:
            logger.info("Loading the stored pre-trained model..")
            trained_model_path = os.path.join(self.pre_trained_model, 'model.pt')
            saved_state = torch.load(trained_model_path,
                                     map_location=torch.device(self.device))  # loading the saved state

            model.load_state_dict(saved_state['model'])  # loading the saved model
            model = model.to(self.device)

            #optimizer.load_state_dict(saved_state['optimizer'])  # loading the saved optimizer
            best_validation_f1 = saved_state['best_validation_f1']  # loading the saved validation f1
            best_test_f1 = saved_state['best_test_f1']  # loading the saved test f1
            del saved_state

            logger.info("Train stored model through incremental contrastive learning.")
            if self.dl_config["MAML"]:
                train_ICL_MAML(icl_old_loader,
                    train_icl_old_dataset,
                    icl_new_loader,
                    train_icl_new_dataset,
                    train_loader,
                    test_loader,
                    valid_loader,
                    model,
                    optimizer_new, 
                    scheduler_new,
                    optimizer_f,
                    scheduler_f,
                    best_test_f1,
                    best_validation_f1,
                    self.dl_config,
                    dl_error_csv_obj,
                    train_error_csv_obj,
                    i
                )
            else:
                train_ICL(icl_old_loader,
                    train_icl_old_dataset,
                    icl_new_loader,
                    train_icl_new_dataset,
                    train_loader,
                    test_loader,
                    valid_loader,
                    model,
                    optimizer_old, 
                    scheduler_old, 
                    optimizer_new, 
                    scheduler_new,
                    optimizer_f,
                    scheduler_f,
                    best_test_f1,
                    best_validation_f1,
                    self.dl_config,
                    dl_error_csv_obj,
                    train_error_csv_obj,
                    i
                )
            
            if model_flag == 1:
                
                # Stopping the flask-app2
                exit_code = subprocess.Popen("./../classifier/shell_script/app2_stop.sh", shell=True,
                                                stdout=subprocess.PIPE)
                subprocess_return = exit_code.stdout.read()
                logger.info(subprocess_return)
                
                # Copying the trained model to /models/model2/
                start_time = time.time()
                destination = self.classifier_config["model_path_2"]
                logger.info(f"Copying the model to {destination}.")
                shutil.copy(
                    os.path.join(self.pre_trained_model, 'model.pt'),
                    os.path.join(destination, 'model.pt')
                )
                logger.info(f"Copied the trained model to {destination} folder.")
                end_time = time.time()
                logger.info(f"Time taken to copy the model to destination folder: {end_time - start_time}")
                
                # Starting the flask-app2
                exit_code = subprocess.Popen("./../classifier/shell_script/app2_start.sh", shell=True,
                                                stdout=subprocess.PIPE)
                logger.info("Sleep for 10s")
                time.sleep(10)
                subprocess_return = exit_code.stdout.read()
                logger.info(subprocess_return)
                
                # Updating the MODEL_FLAG to 2.
                self.model_flag.update_flag(2)
                logger.info("Updated the model flag to 2.")
                logger.info("Sleep for 60s")
                time.sleep(60)
            elif model_flag == 2:
                
                # Stop the flask-app1
                exit_code = subprocess.Popen("./../classifier/shell_script/app1_stop.sh", shell=True,
                                                stdout=subprocess.PIPE)
                subprocess_return = exit_code.stdout.read()
                logger.info(subprocess_return)
                
                # Copy the trained model to FOLDER1
                start_time = time.time()
                logger.info(f"Copying the model to deployment folder.")
                destination = self.classifier_config["model_path_1"]
                shutil.copy(
                    os.path.join(self.pre_trained_model, 'model.pt'),
                    os.path.join(destination, 'model.pt')
                )
                logger.info(f"Copied the trained model to {destination} folder.")
                end_time = time.time()
                logger.info(f"Time taken to copy model to destination folder: {end_time - start_time}")
                
                # Start the flask-app1
                exit_code = subprocess.Popen("./../classifier/shell_script/app1_start.sh", shell=True,
                                                stdout=subprocess.PIPE)
                logger.info("Sleep for 10s")
                time.sleep(10)
                subprocess_return = exit_code.stdout.read()
                logger.info(subprocess_return)
                
                # Change the MODEL_FLAG
                self.model_flag.update_flag(1)
                logger.info("Updated the model flag to 1.")
                logger.info("Sleep for 60s")
                time.sleep(60)
                                
        elif model_flag == 0:
            logger.info("Loading the stored pre-trained model..")
            trained_model_path = os.path.join(self.pre_trained_model, 'model.pt')
            saved_state = torch.load(trained_model_path,
                                     map_location=torch.device(self.device))  # loading the saved state

            model.load_state_dict(saved_state['model'])  # loading the saved model
            model = model.to(self.device)

            #optimizer.load_state_dict(saved_state['optimizer'])  # loading the saved optimizer
            best_validation_f1 = saved_state['best_validation_f1']  # loading the saved validation f1
            best_test_f1 = saved_state['best_test_f1']  # loading the saved test f1
            del saved_state
            
            train(
                trainset_nolabel_loader,
                train_loader,
                test_loader,
                valid_loader,
                model,
                optimizer,
                scheduler,
                optimizer_f, 
                scheduler_f,
                best_test_f1,
                best_validation_f1,
                self.dl_config,
                dl_error_csv_obj,
                train_error_csv_obj,
                i
            )

            start_time = time.time()
            logger.info(f"Copying the model to FOLDER1.")
            destination = self.classifier_config["model_path_1"]
            shutil.copy(os.path.join(self.pre_trained_model, 'model.pt'),
                        os.path.join(destination, 'model.pt'))
            logger.info(f"Copied the trained model to {destination} folder.")
            end_time = time.time()
            logger.info(f"Time taken to copy model to destination folder: {end_time - start_time}")
            
            exit_code = subprocess.Popen("./../classifier/shell_script/app1_start.sh", shell=True,
                                         stdout=subprocess.PIPE)
            logger.info("Sleep for 10s")
            time.sleep(10)
            subprocess_return = exit_code.stdout.read()
            logger.info(subprocess_return)
            
            self.model_flag.update_flag(1)
            logger.info("Updated the MODEL_FLAG to 1.")
            
            start_time = time.time()
            logger.info(f"Copying the model to FOLDER2.")
            destination = self.classifier_config["model_path_2"]
            shutil.copy(os.path.join(self.pre_trained_model, 'model.pt'),
                        os.path.join(destination, 'model.pt'))
            logger.info(f"Copied the trained model to {destination} folder.")
            end_time = time.time()
            logger.info(f"Time taken to copy model to destination folder: {end_time - start_time}")
            
            exit_code = subprocess.Popen("./../classifier/shell_script/app2_start.sh", shell=True,
                                         stdout=subprocess.PIPE)
            logger.info("Sleep for 60s")
            time.sleep(60)
            subprocess_return = exit_code.stdout.read()
            logger.info(subprocess_return)
            

if __name__ == "__main__":
    em = EMStreamTrain()
    