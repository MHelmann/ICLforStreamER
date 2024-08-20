import datetime
import os
import time
import torch
import logging
import pickle

from torch.cuda import amp
from utils.deep_learning_util import train_step, \
    train_icl_step, \
    train_icl_maml_step, \
    fine_tune_step, \
    update_ewc, \
    evaluate

from meta_data.metadata import MetaData
m_data = MetaData()
metaData = m_data.get_meta_data()

logger = logging.getLogger(__name__)


def evaluate_model(model,
                   validate_loader,
                   test_loader,
                   train_loss,
                   dl_error_csv_obj,
                   save_model,
                   model_path,
                   optimizer,
                   best_test_f1,
                   best_validation_f1,
                   i,
                   epoch):
    """
    Evaluate the model on validation and test datasets, and save the model if it achieves a new best validation F1 score.

    Args:
        model (nn.Module): DL model to evaluate.
        validate_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        train_loss (float): Loss value from the training step.
        dl_error_csv_obj: A csv-file to store DL-errors per epoch.
        save_model (bool): Whether to save the model checkpoints or not.
        model_path (str): Path to save the model checkpoints.
        optimizer: Optimizer used during training.
        best_test_f1 (float): Best F1 score achieved on the test dataset so far.
        best_validation_f1 (float): Best F1 score achieved on the validation dataset so far.
        i (int): Current iteration idx.
        epoch (int): Current epoch number.

    Returns:
        None
    """

    timestamp = datetime.datetime.utcnow()
    model.eval()
    valid_loss, valid_eval_metric_list, valid_f_score, th = evaluate(model, validate_loader)
    test_loss, test_eval_metric_list, test_f_score, test_th = evaluate(model, test_loader, threshold=th)

    test_eval_metric_list.append(train_loss)
    test_eval_metric_list.append(valid_loss)
    test_eval_metric_list.append(test_loss)
    test_eval_metric_list.append(timestamp)
    test_eval_metric_list.append(i)
    test_eval_metric_list.append(epoch)

    dl_error_csv_obj.writerow(test_eval_metric_list)

    logger.info(f"MODEL_METRICS: {test_eval_metric_list}")

    if valid_f_score > best_validation_f1:
        best_validation_f1 = valid_f_score
        best_test_f1 = test_f_score
        if save_model:
            logger.info("Saving the model.")

            if not os.path.exists(model_path):
                os.makedirs(model_path)

            check_path = os.path.join(model_path, 'model.pt')
            check_point = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_test_f1': best_test_f1,
                'best_validation_f1': best_validation_f1,
                'test_loss': test_loss,
                'threshold': th
            }
            torch.save(check_point, check_path)

    info = f"epoch {epoch}: validation_loss={valid_loss}, test_loss={test_loss}, train_loss={train_loss}"
    logger.info(info)

    info = f"epoch {epoch}: valid_f1={valid_f_score}, f1={test_f_score}, best_f1={best_test_f1}"
    logger.info(info)

def train(train_ssl_loader,
          train_loader,
          test_loader,
          validate_loader,
          model,
          optimizer,
          scheduler,
          optimizer_f, 
          scheduler_f,
          best_test_f1,
          best_validation_f1,
          config,
          dl_error_csv_obj,
          train_error_csv_obj,
          i):
    """
    Train using Contrastive Learning and evaluate the model.

    Args:
        train_ssl_loader (DataLoader): DataLoader for SSL train data.
        train_loader (DataLoader): DataLoader for the main train data.
        test_loader (DataLoader): DataLoader for the test data.
        validate_loader (DataLoader): DataLoader for the validation data.
        model (nn.Module): Pre-trained language model.
        optimizer: Optimizer used for training.
        scheduler: Scheduler for adjusting the learning rate.
        optimizer_f: Optimizer used for fine-tuning.
        scheduler_f: Scheduler for fine-tuning.
        best_test_f1 (float): Best F1 score achieved on the test dataset so far.
        best_validation_f1 (float): Best F1 score achieved on the validation dataset so far.
        config: Deep-learning configuration object.
        dl_error_csv_obj: A csv-file to store DL-errors per epoch.
        train_error_csv_obj: A csv-file to store DL-train errors.
        i (int): Current iteration number.

    Returns:
        None
    """
    if config["clustering"]:
        n_ssl_epochs = 1
    else:
        n_ssl_epochs = config["SSL_EPOCHS"]
    n_epochs = config["EPOCHS"]
    save_model = config["save_model"]
    model_path = metaData["PRE_TRAINED_MODEL_PATH"]
    ewc_pickle_path = config["ewc_pickle_path"]
    try:
        with open(ewc_pickle_path, 'rb') as file:
            ewc_dict = pickle.load(file)
            logger.info(f"Loaded EWC dictionary.")
    except FileNotFoundError:
        ewc_dict = {"fisher_dict": {}, "optpar_dict": {}}
        logger.info(f"No EWC dictionary found, creating an empty one.")
    except IOError:
        logger.info("Error opening pickle file.")
    
    scaler = amp.GradScaler()
    logger.info("Started training the DL-Model.")
    start_time = time.time()
    # self-supervised learning
    for epoch in range(1, n_ssl_epochs + 1):
        store_epoch = epoch + i
        model.train()
        logger.info(f"SSL EPOCH: {epoch}/{n_ssl_epochs}")
        logger.info("-" * 30)
        train_ssl_loss = train_step(train_ssl_loader,
                                model,
                                optimizer,
                                scheduler,
                                scaler,
                                train_error_csv_obj,
                                store_epoch)

    # fine-tuning with labeled entity pairs
    for epoch in range(1, n_epochs + 1):
        store_epoch = n_ssl_epochs + epoch + i
        logger.info(f"Fine-Tune EPOCH: {epoch}/{n_epochs}")
        logger.info("-" * 30)
        # fine tune step
        model.train()

        train_loss = fine_tune_step(train_loader,
                                    model,
                                    optimizer_f,
                                    scheduler_f,
                                    scaler,
                                    train_error_csv_obj,
                                    store_epoch,
                                    ewc_dict,
                                    i)
        
        # evalutaion
        evaluate_model(model,
                       validate_loader,
                       test_loader,
                       train_loss,
                       dl_error_csv_obj,
                       save_model,
                       model_path,
                       optimizer,
                       best_test_f1,
                       best_validation_f1,
                       i,
                       epoch)
    
    ewc_dict = update_ewc(train_loader, model, optimizer, ewc_dict, i)
    with open(ewc_pickle_path, "wb") as file:
        pickle.dump(ewc_dict, file)
    info = f"Loop {i}: Updated EWC parameter dictionary and saved to ewc.pickle."
    logger.info(info)

    end_time = time.time()
    logger.info(f"Time taken for training the model: {end_time - start_time}")


def train_ICL(icl_old_loader,
            train_icl_old_dataset,
            icl_new_loader,
            train_icl_new_dataset,
            train_loader,
            test_loader,
            validate_loader,
            model,
            optimizer_old, 
            scheduler_old, 
            optimizer_new, 
            scheduler_new,
            optimizer_f,
            scheduler_f,
            best_test_f1,
            best_validation_f1,
            config,
            dl_error_csv_obj,
            train_error_csv_obj,
            i):
    """
    Train using Incremental Contrastive Learning (ICL) and evaluate the model.

    Args:
        icl_old_loader (DataLoader): DataLoader for old data for ICL training.
        train_icl_old_dataset: Old data for ICL training (dataset).
        icl_new_loader (DataLoader): DataLoader for new data for ICL training.
        train_icl_new_dataset: New data for ICL training (dataset).
        train_loader (DataLoader): DataLoader for the overall training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        validate_loader (DataLoader): DataLoader for the validation dataset.
        model (nn.Module): Pre-trained language model.
        optimizer_old: Optimizer for old data in ICL training.
        scheduler_old: Scheduler for old data in ICL training.
        optimizer_new: Optimizer for new data in ICL training.
        scheduler_new: Scheduler for new data in ICL training.
        optimizer_f: Optimizer for fine-tuning.
        scheduler_f: Scheduler for fine-tuning.
        best_test_f1 (float): Best F1 score achieved on the test dataset so far.
        best_validation_f1 (float): Best F1 score achieved on the validation dataset so far.
        config: Deep-learning configuration object.
        dl_error_csv_obj: A csv-file to store DL-errors per epoch.
        train_error_csv_obj: A csv-file to store DL-train errors.
        i (int): Current iteration number.

    Returns:
        None
    """
    if config["clustering"]:
        n_icl_epochs = 1
    else:
        n_icl_epochs = config["ICL_EPOCHS"]
    n_epochs = config["EPOCHS"] # fine-tune
    save_model = config["save_model"]
    model_path = metaData["PRE_TRAINED_MODEL_PATH"]
    ewc_pickle_path = config["ewc_pickle_path"]
    try:
        with open(ewc_pickle_path, 'rb') as file:
            ewc_dict = pickle.load(file)
            logger.info(f"Loaded EWC dictionary.")
    except FileNotFoundError:
        ewc_dict = {"fisher_dict": {}, "optpar_dict": {}}
        logger.info(f"No EWC dictionary found, creating an empty one.")
    except IOError:
        print("Error opening pickle file.")
    
    scaler = amp.GradScaler()
    logger.info("Started ICL training of DL-Model.")
    start_time = time.time()
    for epoch in range(1, n_icl_epochs + 1):
        store_epoch = epoch + i
        logger.info(f"ICL EPOCH: {epoch}/{n_icl_epochs}")
        logger.info("-" * 30)
        model.train()
        train_icl_loss = train_icl_step(icl_old_loader,
                                train_icl_old_dataset,
                                icl_new_loader,
                                train_icl_new_dataset,
                                model,
                                optimizer_old,
                                scheduler_old,
                                optimizer_new,
                                scheduler_new,
                                scaler,
                                train_error_csv_obj,
                                store_epoch)
    
    # fine-tuning with labeled entity pairs
    for epoch in range(1, n_epochs + 1):
        store_epoch = n_icl_epochs + epoch + i
        logger.info(f"Fine-Tune EPOCH: {epoch}/{n_epochs}")
        logger.info("-" * 30)
        # fine tune step
        model.train()

        train_loss = fine_tune_step(train_loader,
                                    model,
                                    optimizer_f,
                                    scheduler_f,
                                    scaler,
                                    train_error_csv_obj,
                                    store_epoch,
                                    ewc_dict,
                                    i)
        # evalutaion
        evaluate_model(model,
                       validate_loader,
                       test_loader,
                       train_loss,
                       dl_error_csv_obj,
                       save_model,
                       model_path,
                       optimizer_f,
                       best_test_f1,
                       best_validation_f1,
                       i,
                       epoch)
        
    ewc_dict = update_ewc(train_loader, model, optimizer_f, ewc_dict, i)
    with open(ewc_pickle_path, "wb") as file:
        pickle.dump(ewc_dict, file)
    info = f"Loop {i}: Updated EWC parameter dictionary and saved to ewc.pickle."
    logger.info(info)
    
    end_time = time.time()
    logger.info(f"Time taken for training the model: {end_time - start_time}")

def train_ICL_MAML(icl_old_loader,
            train_icl_old_dataset,
            icl_new_loader,
            train_icl_new_dataset,
            train_loader,
            test_loader,
            validate_loader,
            model,
            optimizer_new,
            scheduler_new,
            optimizer_f,
            scheduler_f,
            best_test_f1,
            best_validation_f1,
            config,
            dl_error_csv_obj,
            train_error_csv_obj,
            i):
    """
    Train using Incremental Contrastive Learning (ICL) with Model-Agnostic Meta-Learning (MAML) and evaluate the model.

    Args:
        icl_old_loader (DataLoader): DataLoader for old data for ICL-MAML training.
        train_icl_old_dataset: Old data for ICL-MAML training (dataset).
        icl_new_loader (DataLoader): DataLoader for new data for ICL-MAML training.
        train_icl_new_dataset: New data for ICL-MAML training (dataset).
        train_loader (DataLoader): DataLoader for the overall training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        validate_loader (DataLoader): DataLoader for the validation dataset.
        model (nn.Module): Pre-trained language model.
        optimizer_new: Optimizer for new data in ICL-MAML training.
        scheduler_new: Scheduler for new data in ICL-MAML training.
        optimizer_f: Optimizer for fine-tuning.
        scheduler_f: Scheduler for fine-tuning.
        best_test_f1 (float): Best F1 score achieved on the test dataset so far.
        best_validation_f1 (float): Best F1 score achieved on the validation dataset so far.
        config: Deep-learning configuration object.
        dl_error_csv_obj: A csv-file to store DL-errors per epoch.
        train_error_csv_obj: A csv-file to store DL-train errors.
        i (int): Current iteration number.

    Returns:
        None
    """
    if config["clustering"]:
        n_icl_epochs = 1
    else:
        n_icl_epochs = config["ICL_EPOCHS"]
    n_epochs = config["EPOCHS"] # fine-tune
    save_model = config["save_model"]
    model_path = metaData["PRE_TRAINED_MODEL_PATH"]
    ewc_pickle_path = config["ewc_pickle_path"]
    try:
        with open(ewc_pickle_path, 'rb') as file:
            ewc_dict = pickle.load(file)
            logger.info(f"Loaded EWC dictionary.")
    except FileNotFoundError:
        ewc_dict = {"fisher_dict": {}, "optpar_dict": {}}
        logger.info(f"No EWC dictionary found, creating an empty one.")
    except IOError:
        print("Error opening pickle file.")
   
    scaler = amp.GradScaler()
    logger.info("Started ICL-MAML training of DL-Model.")
    start_time = time.time()
    
    optimizer_inner = torch.optim.AdamW(model.parameters(), lr=4.5e-5)
    for epoch in range(1, n_icl_epochs + 1):
        store_epoch = epoch + i
        logger.info(f"ICL-MAML EPOCH: {epoch}/{n_icl_epochs}")
        logger.info("-" * 30)
        model.train()
        train_icl_maml_loss = train_icl_maml_step(icl_old_loader,
                                train_icl_old_dataset,
                                icl_new_loader,
                                train_icl_new_dataset,
                                model,
                                optimizer_new,
                                optimizer_inner,
                                scheduler_new,
                                scaler,
                                train_error_csv_obj,
                                store_epoch)

    # fine-tuning with labeled entity pairs
    for epoch in range(1, n_epochs + 1):
        store_epoch = n_icl_epochs + epoch + i
        logger.info(f"Fine-Tune EPOCH: {epoch}/{n_epochs}")
        logger.info("-" * 30)
        # fine tune step
        model.train()

        train_loss = fine_tune_step(train_loader,
                                    model,
                                    optimizer_f,
                                    scheduler_f,
                                    scaler,
                                    train_error_csv_obj,
                                    store_epoch,
                                    ewc_dict,
                                    i)
        # evalutaion
        evaluate_model(model,
                       validate_loader,
                       test_loader,
                       train_loss,
                       dl_error_csv_obj,
                       save_model,
                       model_path,
                       optimizer_f,
                       best_test_f1,
                       best_validation_f1,
                       i,
                       epoch)
        
    ewc_dict = update_ewc(train_loader, model, optimizer_f, ewc_dict, i)
    with open(ewc_pickle_path, "wb") as file:
        pickle.dump(ewc_dict, file)
    info = f"Loop {i}: Updated EWC parameter dictionary and saved to ewc.pickle."
    logger.info(info)
    
    end_time = time.time()
    logger.info(f"Time taken for training the model: {end_time - start_time}")

