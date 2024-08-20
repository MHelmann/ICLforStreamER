import time
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
from sklearn.metrics import classification_report, f1_score
import logging
from tqdm import tqdm
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup
from torch.cuda import amp
from DLEM.model import EMModel
from DLEM.criterion import info_nce_loss, loss_cal_ICL_old, loss_cal_ICL_new
from DLEM.prepare_dataset import EMDataset, SSLDataset
from config.config import get_deep_learning_config

dl_config = get_deep_learning_config()

logger = logging.getLogger(__name__)


def evaluation_metrices(true_label, pred_label):
    eval_metric = classification_report(true_label, pred_label, output_dict=True, zero_division=1)
    eval_metric_list = [eval_metric['0']['precision'],
                        eval_metric['0']['recall'],
                        eval_metric['0']['f1-score'],
                        eval_metric['0']['support'],
                        eval_metric['1']['precision'],
                        eval_metric['1']['recall'],
                        eval_metric['1']['f1-score'],
                        eval_metric['1']['support'],
                        eval_metric['accuracy'],
                        eval_metric['macro avg']['precision'],
                        eval_metric['macro avg']['recall'],
                        eval_metric['macro avg']['f1-score'],
                        eval_metric['macro avg']['support'],
                        eval_metric['weighted avg']['precision'],
                        eval_metric['weighted avg']['recall'],
                        eval_metric['weighted avg']['f1-score'],
                        eval_metric['weighted avg']['support']
                        ]

    return eval_metric_list


def evaluate(model, iterator, threshold=None):
    """
    This function evaluates the deep learning model.
    :param model: pre-trained language model.
    :param iterator: entity_pool-loader
    :param threshold: Threshold to classify the entity pairs.
    :return : f1-score, mean-loss, best threshold.
    """
    all_y = []
    all_probs = []
    losses = []
    eval_metric_list = []
    criterion = nn.CrossEntropyLoss()
    i = 0
    with torch.no_grad():
        for batch in tqdm(iterator):
            y1, y2, y12, labels = batch
            logits, embeddings = model(2, y1, y2, y12)
            loss = criterion(logits, labels.to(model.device))

            losses.append(loss.item())

            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += labels.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        # f1 = metrics.f1_score(all_y, pred, zero_division=1)
        eval_metric_list = evaluation_metrices(all_y, pred)
        f_score = f1_score(all_y, pred, zero_division=1)
        # return np.mean(losses), eval_metric_list, f_score, threshold
        best_th = threshold
    else:
        best_th = 0.5
        f_score = 0.0
        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f_score = f1_score(all_y, pred, zero_division=1)
            if new_f_score > f_score:
                f_score = new_f_score
                best_th = th
                eval_metric_list = evaluation_metrices(all_y, pred)

    return np.mean(losses), eval_metric_list, f_score, best_th


def get_ssl_dataset(X):
    """
    This function creates dataset for Contrastive Learning with respect to used language-model.
    :param X: List of serialized entity pairs.
    :return : The language-model's dataset object.
    """
    dataset = SSLDataset(
        X,
        dl_config["MAX_LEN"],
        dl_config["size"],
        dl_config["lm"],
        dl_config["da_ssl"]
    )
    return dataset

def get_dataset(X, y):
    """
    This function creates dataset with respect to language-model used.
    :param X: List of serialized entity pairs.
    :param y: List of corresponding labels.
    :return : Language-model's dataset object.
    """
    dataset = EMDataset(
        X,
        y,
        dl_config["MAX_LEN"],
        dl_config["lm"]
    )
    return dataset


def get_data_loader(flag, dataset, padder, train=None):
    """
    This function converts language-model dataset to dataloader object.
    :param dataset: the language-model dataset object
    :param padder: The padding object to pad tokens to max-length.
    :param train: If "train", then use half the batch size.
    :return : Dataloader object.
    """
    if flag == 0:
        batch_size=dl_config["BATCH_SIZE"]
    elif flag == 1:
        batch_size=dl_config["BATCH_SIZE_SSL"]

    if train == "train":
        batch_size = batch_size//2

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=padder
    )

    return data_loader


def get_model(device):
    """
    This function creates a model of given language-model (Distil-BERT, RoBERTa).
    :param device: Processor on which the deep-learning calculation occurs.
    :return : Language-model object.
    """
    try:
        model = EMModel(device, dl_config["lm"])
    except Exception as arg:
        logger.info("Exception while creating the DL-model.")
        logger.info(arg)
    else:
        return model


def get_ssl_optimizer_scheduler(dataset_ssl, model):
    """
    This function creates optimizer and scheduler of deep learning model for the 1st iteration 
    where only Contrastive Learning is used.
    :param dataset: Input datasets to calculate number of scheduler steps during Contrastive Learning.
    :param model: pre-trained language model (Distil-BERT, RoBERTa)
    """
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=dl_config["lr"])
        num_ssl_epochs = dl_config["SSL_EPOCHS"]
        num_ssl_steps = len(dataset_ssl) // dl_config["BATCH_SIZE_SSL"] * num_ssl_epochs
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_ssl_steps,
                                                lr_end=3.8e-5,
                                                power=1.0)
    except Exception:
        logger.exception("Exception while creating optimizer and scheduler.", exc_info=True)
        raise
    else:
        return optimizer, scheduler
    

def get_icl_optimizer_scheduler(icl_old_dataset, icl_new_dataset, model):
    """
    This function creates optimizer and scheduler of deep learning model 
    for Incremental Contrastive Learning.
    :param dataset: Input datasets to calculate number of scheduler steps during Incremental Contrastive Learning.
    :param model: pre-trained language model (Distil-BERT, RoBERTa)
    """
    try:
        num_icl_epochs = dl_config["ICL_EPOCHS"]
        if icl_old_dataset: # Gets initialized if normal ICL without MAML is used.
            optimizer_old = torch.optim.AdamW(model.parameters(), lr=dl_config["lr"])
            num_icl_old_steps = len(icl_old_dataset) // dl_config["BATCH_SIZE_SSL"] * num_icl_epochs
            scheduler_old = get_polynomial_decay_schedule_with_warmup(optimizer_old,
                                                num_warmup_steps=0,
                                                num_training_steps=num_icl_old_steps,
                                                lr_end=3.5e-5,
                                                power=1.0)
        
        optimizer_new = torch.optim.AdamW(model.parameters(), lr=dl_config["lr"])
        num_icl_new_steps = len(icl_new_dataset) // dl_config["BATCH_SIZE_SSL"] * num_icl_epochs
        scheduler_new = get_polynomial_decay_schedule_with_warmup(optimizer_new,
                                                num_warmup_steps=0,
                                                num_training_steps=num_icl_new_steps,
                                                lr_end=3.5e-5,
                                                power=1.0)
    except Exception:
        logger.exception("Exception while creating optimizer and scheduler.", exc_info=True)
        raise
    else:
        if icl_old_dataset:
            return optimizer_old, scheduler_old, optimizer_new, scheduler_new
        else:
            return optimizer_new, scheduler_new


def get_optimizer_scheduler(trainset, model):
    """
    This function creates optimizer and scheduler of deep learning for fine-tuning.
    :param dataset: Input datasets to calculate number of steps for fine-tuning steps in scheduler.
    :param model: pre-trained language model (Distil-BERT, RoBERTa)
    """
    try:
        optimizer = AdamW(model.parameters(), lr=3.5e-5)
        num_steps = (len(trainset) // dl_config["BATCH_SIZE"] // 2) * dl_config["EPOCHS"]
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps,
                                                lr_end=2.5e-5,
                                                power=1.0)
    except Exception:
        logger.exception("Exception while creating optimizer and scheduler.", exc_info=True)
        raise
    else:
        return optimizer, scheduler


def serialise_entity_pair_ssl(entity_a, entity_b):
    """
    This function serializes the entities of dictionary type to string
    that are used for the self-supervised training.

    :param entity_a: Entity-a dictionary.
    :param entity_b: Entity-b dictionary.
    :return : list consisting serialized entity pair.
    """
    try:
        entities = [entity_a, entity_b]
        entity_pairs = []
        for entity in entities:
            entity_series = ""
            for key, val in entity.items():
                if key != "_id":
                    key = key.replace("\n", "")
                    val = val.replace("\n", "")

                    key = key.replace("\t", " ")
                    val = val.replace("\t", " ")

                    entity_series += f"COL {key} VAL {val} "
            entity_series = entity_series.strip()
            entity_pairs.append(entity_series)
        return entity_pairs
    except Exception:
        logger.exception("Exception while serializing the entity pairs for SSL.", exc_info=True)
        raise


def serialization_ssl(entities):
    """
    This function serializes the entity pairs from dictionary to strings
    that are used for the self-supervised training.
    :param entities: Entity consisting entity-a, entity-b
    :return : List consisting serialized entities.
    """
    entity_pairs = []
    logger.info("Entity pair serialization for SSL started...")
    start_time = time.time()
    for entity in entities:
        entity_a = entity['entity_a']
        entity_b = entity['entity_b']

        serial_entity_pairs = serialise_entity_pair_ssl(entity_a,
                                                    entity_b)

        ent = {
            0: serial_entity_pairs[0],
            1: serial_entity_pairs[1]
        }
        entity_pairs.append(ent)
    end_time = time.time()
    logger.info("Entity pair serialization for SSL completed.")
    logger.info(f"Time taken for SSL serialization: {end_time - start_time}")
    return entity_pairs


def serialise_entity_pair(entity_a, entity_b, weight, label=None):
    """
    This function serializes the entities of dictionary type to string.

    :param entity_a: Entity-a dictionary.
    :param entity_b: Entity-b dictionary.
    :param label: the true label of entity-pairs.
    :return : list consisting serialized entity pair.
    """
    try:
        entities = [entity_a, entity_b]
        entity_pairs = []
        for entity in entities:
            entity_series = ""
            for key, val in entity.items():
                if key != "_id":
                    key = key.replace("\n", "")
                    val = val.replace("\n", "")

                    key = key.replace("\t", " ")
                    val = val.replace("\t", " ")

                    entity_series += f"COL {key} VAL {val} "
            entity_series = entity_series.strip()
            entity_pairs.append(entity_series)
        if label is None:
            return entity_pairs
        else:
            label = str(label)
            entity_pairs.append(label)
            # entity_serialized = '\t'.join(entity_pairs)
            return entity_pairs
    except Exception:
        logger.exception("Exception while serializing the entity pairs.", exc_info=True)
        raise


def serialization(entities):
    """
    This function serializes the entity pairs from dictionary to strings.
    :param entities: Entity consisting entity-a, entity-b and true-label.
    :return : List consisting serialized entities.
    """
    entity_pairs = []
    logger.info("Entity pair serialization for fine-tuning started...")
    start_time = time.time()
    for entity in entities:
        entity_a = entity['entity_a']
        entity_b = entity['entity_b']
        weight = entity['weight']
        label = entity['label_o']

        serial_entity_pairs = serialise_entity_pair(entity_a,
                                                    entity_b,
                                                    weight,
                                                    label)
        ent = {
            0: serial_entity_pairs[0],
            1: serial_entity_pairs[1],
            2: int(serial_entity_pairs[2])
        }
        entity_pairs.append(ent)
    end_time = time.time()
    logger.info("Entity pair serialization completed.")
    logger.info(f"Time taken for serialization: {end_time - start_time}")
    return entity_pairs


def create_batches(u_set, batch_size, n_ssl_epochs, num_clusters=90):
    """
    Generate batches such that similar entries are grouped together.
    Aim is to obtain more effective negative samples for Contrastive Learning.
    :param u_set: Language-model's dataset object.
    :param batch_size: Size of batches to be created for training.
    :param n_ssl_epochs: Number of epochs for Contrastive Learnin
    :param num_clusters: Number of clusters for k-Means
    :return : Dataloader object.
    """
    batch_size = batch_size
    n_ssl_epochs = n_ssl_epochs
    N = len(u_set)
    tfidf = TfidfVectorizer().fit_transform(u_set.instances)

    kmeans = KMeans(n_clusters=num_clusters).fit(tfidf)

    clusters = [[] for _ in range(num_clusters)]
    for idx, label in enumerate(kmeans.labels_):
        clusters[label].append(idx)

    # concatenate
    batch_list = []
    for _ in range(n_ssl_epochs):
        indices = []
        random.shuffle(clusters)

        for c in clusters:
            random.shuffle(c)
            indices += c

        batch = []
        for i, idx in enumerate(indices):
            batch.append(u_set[i])
            if len(batch) == batch_size or i == N - 1:
                batch_list.append(u_set.pad(batch))
                batch.clear()

    # Create DataLoader
    dataloader = data.DataLoader(batch_list,
                                batch_size=None,
                                shuffle=False,
                                num_workers=0
                                )

    return dataloader


def train_step(train_loader,
               model,
               optimizer,
               scheduler,
               scaler,
               train_error_csv_obj,
               store_epoch):
    """
    Perform a single Contrastive Learning trainig step
    Adapted from https://github.com/megagonlabs/sudowoodo/tree/main

    :param train_loader (Iterator): Iterator over the dataloader
    :param model (DMModel): DL model used for prediction
    :param optimizer (Optimizer): Optimizer (Adam or AdamW)
    :param scheduler (LRScheduler): Learning rate scheduler
    :param scaler: Gradient scaling to mitigate potential underflow or overflow issues
    :param train_error_csv_obj: A csv-file to store DL-train errors.
    :param store_epoch: Current epoch of training
    :returns : Loss
    """
    losses = []
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        with amp.autocast():
            yA, yB = batch
            encoding = model(0, yA, yB, [], da='cutoff', cutoff_ratio=0.05)
            logits, labels = info_nce_loss(model.device, features=encoding, batch_size=len(yA), n_views=2)
            loss1 = criterion(logits, labels)

            alpha = 1 - 0.001
            loss2 = model(1, yA, yB, [], da='cutoff', cutoff_ratio=0.05)

            loss = alpha * loss1 + (1 - alpha) * loss2
        
        losses.append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()

        if i % 10 == 0:
            train_error_csv_obj.writerow([store_epoch, i, loss.item()])
            logger.info(f"step: {i}, ssl-loss: {loss.item()}")
        del loss
        
    return np.mean(losses)


def train_icl_step(icl_old_loader,
                icl_old_dataset,
                icl_new_loader,
                icl_new_dataset,
                model,
                optimizer_old,
                scheduler_old,
                optimizer_new,
                scheduler_new,
                scaler,
                train_error_csv_obj,
                store_epoch):
    """
    Perform a single ICL trainig step
    Adapted from https://github.com/RingBDStack/ICL-Incremental-InfoNCE/tree/main

    :param icl_old_loader (Iterator): Iterator over the dataloader for old samples
    :param icl_old_dataset (Iterator): Dataset containing old samples
    :param icl_new_loader (Iterator): Iterator over the dataloader for new samples
    :param icl_new_dataset (Iterator): Dataset containing new samples
    :param model (DMModel): DL model used for prediction
    :param optimizer_old (Optimizer): Optimizer (Adam or AdamW) for old data
    :param scheduler_old (LRScheduler): Learning rate scheduler
    :param optimizer_new (Optimizer): Optimizer (Adam or AdamW) for new data
    :param scheduler_new (LRScheduler): Learning rate scheduler
    :param scaler: Gradient scaling to mitigate potential underflow or overflow issues
    :param train_error_csv_obj: A csv-file to store DL-train errors.
    :param store_epoch: Current epoch of training
    :returns : Loss
    """
    alpha = len(icl_new_dataset) / (len(icl_new_dataset) + len(icl_old_dataset))
    losses = []
    for i, old_batch in enumerate(icl_old_loader):
        optimizer_old.zero_grad()
        with amp.autocast():
            yA, yB = old_batch
            encoding = model(0, yA, yB, [], da=None, cutoff_ratio=0.05)
            old_out = encoding[:len(yA), :]
            old_out_aug = encoding[len(yA):, :]
            
            try:
                new_batch = next(new4old_iter)
            except:
                new4old_iter = iter(create_batches(icl_new_dataset, len(yA)-1, 1))
                new_batch = next(new4old_iter)

            new_yA, new_yB = new_batch
            encoding = model(0, new_yA, new_yB, [], da=None, cutoff_ratio=0.05)    
            new_out = encoding[:len(new_yA)]
            
            loss = loss_cal_ICL_old(old_out, old_out_aug, new_out, alpha)
    
        losses.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer_old)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        if not skip_lr_sched:
            scheduler_old.step()
       
        if i % 10 == 0:
            train_error_csv_obj.writerow([store_epoch, i, loss.item()])
            logger.info(f"step: {i}, ICL-old-loss: {loss.item()}")
        del loss


    for j, new_batch in enumerate(icl_new_loader):
        optimizer_new.zero_grad()
        with amp.autocast():
            yA, yB = new_batch
            batch_size = len(yA)
            old_num = int((1 - alpha) * (batch_size-1))
            new_num = batch_size-1 - old_num

            encoding = model(0, yA, yB, [], da=None, cutoff_ratio=0.05)
            new_out = encoding[:batch_size, :]
            new_out_aug = encoding[batch_size:, :]
            
            if old_num == 0:
                old_out = None
            else:                
                try:
                    old_batch = next(old4new_iter)
                except:
                    old4new_iter = iter(create_batches(icl_new_dataset, old_num, 1))
                    old_batch = next(old4new_iter)
                
                old_yA, old_yB = old_batch
                encoding = model(0, old_yA, old_yB, [], da=None, cutoff_ratio=0.05)    
                old_out = encoding[:len(old_yA)]
            
            loss = loss_cal_ICL_new(new_out, new_out_aug, old_out, new_num)
        
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer_new)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        if not skip_lr_sched:
            scheduler_new.step()

        if j % 10 == 0:
            train_error_csv_obj.writerow([store_epoch, j, loss.item()])
            logger.info(f"step: {j}, ICL-new-loss: {loss.item()}")
        del loss
            
    return np.mean(losses)


def train_icl_maml_step(icl_old_loader,
                icl_old_dataset,
                icl_new_loader,
                icl_new_dataset,
                model,
                optimizer,
                optimizer_inner,
                scheduler,
                scaler,
                train_error_csv_obj,
                store_epoch):
    """
    Perform a single ICL-MAML trainig step
    Adapted from https://github.com/RingBDStack/ICL-Incremental-InfoNCE/tree/main

    :param icl_old_loader (Iterator): Iterator over the dataloader for old samples
    :param icl_old_dataset (Iterator): Dataset containing old samples
    :param icl_new_loader (Iterator): Iterator over the dataloader for new samples
    :param icl_new_dataset (Iterator): Dataset containing new samples
    :param model (DMModel): DL model used for prediction
    :param optimizer (Optimizer): Optimizer (Adam or AdamW)
    :param optimizer_inner (Optimizer): Optimizer (Adam or AdamW)
    :param scheduler (LRScheduler): Learning rate scheduler
    :param scaler: Gradient scaling to mitigate potential underflow or overflow issues
    :param train_error_csv_obj: A csv-file to store DL-train errors.
    :param store_epoch: Current epoch of training
    :returns : Loss
    """
    alpha = len(icl_new_dataset) / (len(icl_new_dataset) + len(icl_old_dataset))

    losses = []
    old_step = max(int(len(icl_old_loader) / len(icl_new_loader)), 1)

    for i, new_batch in enumerate(icl_new_loader):
        # meta-train stage
        for step in range(old_step):
            optimizer.zero_grad()
            optimizer_inner.zero_grad()
            with amp.autocast():
                try:
                    old_batch = next(old_iter)
                except:
                    old_iter = iter(icl_old_loader)
                    old_batch = next(old_iter)

                yA, yB = old_batch
                encoding = model(0, yA, yB, [], da=None, cutoff_ratio=0.1)
                old_out = encoding[:len(yA), :]
                old_out_aug = encoding[len(yA):, :]

                try:
                    new4old_batch = next(new4old_iter)
                except:
                    new4old_iter = iter(create_batches(icl_new_dataset, len(yA)-1, 1))
                    new4old_batch = next(new4old_iter)

                new4old_A, new4old_B = new4old_batch
                encoding = model(0, new4old_A, new4old_B, [], da=None, cutoff_ratio=0.05)
                new4old_out = encoding[:len(new4old_A)]

                loss = loss_cal_ICL_old(old_out, old_out_aug, new4old_out, alpha)

            loss.backward()
            optimizer_inner.step()

        # meta-testing stage
        optimizer.zero_grad()
        with amp.autocast():
            new_batch, new_batch_aug = new_batch
            batch_size = len(new_batch)
            old_num = int((1 - alpha) * (batch_size-1))
            new_num = batch_size - 1 - old_num

            encoding = model(0, new_batch, new_batch_aug, [], da=None, cutoff_ratio=0.05)
            new_out = encoding[:batch_size, :]
            new_out_aug = encoding[batch_size:, :]

            if old_num == 0:
                old_out = None
            else:
                try:
                    old_batch = next(old4new_iter)
                except:
                    old4new_iter = iter(create_batches(icl_old_dataset, old_num, 1))
                    old_batch = next(old4new_iter)

                old_yA, old_yB = old_batch
                encoding = model(0, old_yA, old_yB, [], da=None, cutoff_ratio=0.05)
                old_out = encoding[:len(old_yA)]

            loss = loss_cal_ICL_new(new_out, new_out_aug, old_out, new_num)

        losses.append(loss.item())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
            
        if i % 10 == 0:
            train_error_csv_obj.writerow([store_epoch, i, loss.item()])
            logger.info(f"step: {i}, ICL-MAML-loss: {loss.item()}")
        del loss

    return np.mean(losses)


def fine_tune_step(train_loader,
                   model,
                   optimizer,
                   scheduler,
                   scaler,
                   train_error_csv_obj,
                   store_epoch,
                   ewc_dict,
                   i):
    """
    Perform a single fine-tuning step

    :param train_iter (Iterator): train data loader over labeled dataset
    :param model (DMModel): DL model used for prediction
    :param optimizer (Optimizer): Optimizer (Adam or AdamW)
    :param scheduler (LRScheduler): Learning rate scheduler
    :param scaler: Gradient scaling to mitigate potential underflow or overflow issues
    :param train_error_csv_obj: A csv-file to store DL-train errors.
    :param store_epoch: Current epoch of training
    :param ewc_dict: Dictionary with model parametes of task-specific layer for each epoch
    :param i: Loop idx (task id)
    :returns : Loss
    """
    losses = []
    criterion = nn.CrossEntropyLoss()
    for j, batch in enumerate(train_loader):
        optimizer.zero_grad()
        with amp.autocast():
            y1, y2, y12, labels = batch
            prediction, _ = model(2, y1, y2, y12)
            loss = criterion(prediction, labels.to(model.device))

            # EWC regularization
            for loop_idx in range(i):
                for name, param in model.named_parameters():
                    if "fc" in name:
                        fisher = ewc_dict["fisher_dict"][loop_idx][name]
                        optpar = ewc_dict["optpar_dict"][loop_idx][name]
                        # print(optpar)
                        loss += (fisher * (optpar - param).pow(2)).sum() * dl_config["ewc_lambda"]


        losses.append(loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale != scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
            
        if j % 10 == 0: # monitoring
            train_error_csv_obj.writerow([store_epoch, j, loss.item()])
            logger.info(f"step: {j}, fine-tune loss: {loss.item()}")
        del loss

    return np.mean(losses)


def update_ewc(train_loader,
               model,
               optimizer,
               ewc_dict,
               loop_idx):
    """
    Adapted from: https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
    This function stores weights and their corresponding weights in a dictionary after each
    model pipeline loop. These values are used in the next iteration to adjust the loss, so
    that the model does not forget how to perform on previous tasks.

    :param train_loader: Dataloader over labeled dataset
    :param model (DMModel): DL model used for prediction
    :param optimizer (Optimizer): Optimizer (Adam or AdamW)
    :param ewc_dict: Dictionary to store model parametes of task-specific layer for each epoch
    :param i: Loop idx (task id)
    :return : 
    """
    model.train()
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    # accumulating gradients
    for i, batch in enumerate(train_loader):
        with amp.autocast():
            y1, y2, y12, labels = batch
            prediction = model(1, y1, y2, y12)
            loss = criterion(prediction, labels.to(model.device))  
        loss.backward()

    ewc_dict["fisher_dict"][loop_idx] = {}
    ewc_dict["optpar_dict"][loop_idx] = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        if "fc" in name:
            ewc_dict["optpar_dict"][loop_idx][name] = param.data.clone()
            ewc_dict["fisher_dict"][loop_idx][name] = param.grad.data.clone().pow(2)
    return ewc_dict


def train_test_valid_split(df):
    """
    This function splits the dataframe into train, test, and validation datasets.

    :param df: Pandas dataframe consisting of dataset.
    :return : Train, test, and validation datasets.
    """
    try:
        X = df[[0, 1]].to_numpy()
        y = df[2].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                              shuffle=True)
    except Exception:
        logger.exception("Exception which splitting the dataset into train, test, and valid datasets.", exc_info=True)
        raise
    else:
        logger.info("The dataset is split into train, test, and validation datasets.")
        return X_train, X_test, X_valid, y_train, y_test, y_valid
    

