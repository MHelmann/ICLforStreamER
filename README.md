# [Incremental Contrastive Learning for Entity Resolution on Dynamic Data](DLStreamER.pdf)
This repository contains the source code for the research project: Incremental Contrastive Learning for Entity Resolution on Dynamic Data.

## Abstract
Entity Resolution (ER) problem aims to identify which virtual representations of entities correspond to the same real-world entity. Recently deep learning (DL) based approaches achieved SOTA results in ER on non-dynamic data. However, the research still falls short when applying DL models to dynamic data, also known as the Big Data challenge of velocity. While blocking techniques to cope with this challenge have been proposed, the effectiveness in correctly classifying matches vs non-matches is still lacking. Preliminary work proposed a novel framework that adapts the classification stage to dynamic data. However, the framework still faces a few limitations such as the requirement of high-quality labeled datasets to fine-tune the DL-based model and catastrophic forgetting due to simple retraining on new data.

In this research project, we aim to address the aforementioned limitations by proposing a deep learning-based classification function that is based on Incremental Contrastive representation learning. Contrastive Learning enables the model to learn similarity-aware data representation without using any labels, while the incremental step of the framework guarantees the prevention of bias in the model during the iterative training and as well the alleviation of catastrophic forgetting. Afterward, the learned representations are used to facilitate the fine-tuning of the model to support the classification of streaming entity pairs into match or non-match. The end-to-end system can be divided into two layers; one for iterative training of the model and the other one for the classification task. Both the training and prediction layers work in parallel and independent of each other.

Our experiment results show that the proposed model achieves competitive performance on different benchmark datasets that vary in size and origin domain. In addition, the iterative model performs similarly or even better than the preliminary work.

## Input
All the input files are considered from [JedAI](https://github.com/scify/JedAIToolkit/tree/master/data).

- The input files having entity descriptions and ground truth match pairs are stored in the [entity_pool](input/entity_pool) folder.
- The input file, obtained from the blocking stage, is stored in the [input](input) folder.
- Note: The DBPedia data .json files are stored in the [department cloud](https://ipvscloud.informatik.uni-stuttgart.de/s/kapFcC4stbQNTRG?path=%2F). To produce the triplets from the blocking stage (entity ids and "score") used as input to the devised framework, one needs to run the runProgressiveER.sh file that can be found in the department cloud as well. A detailed procedure on how to execute the .sh file is also included. Important: The input for the runProgressiveER.sh are not DBPedia .json files, those are used for the proposed framework and should be stored in the [entity_pool](input/entity_pool) folder. Instead, download [DBPedia data files](https://deref-gmx.net/mail/client/CRrlpgC4qnM/dereferrer/?redirectUrl=https%3A%2F%2Fdata.mendeley.com%2Fdatasets%2F4whpm32y47%2F7%2Ffiles%2F1a89ef99-7195-449e-95ed-b49bb2a853c4&lm) that are serialized Java objects and use them as input for runProgressiveER.sh.

## Initial Implementations
Kafka:  https://kafka.apache.org/quickstart  
MongoDB: https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-ubuntu/  
Install the [requirements.txt](requirements.txt) file

## [Configuration file](config/config.json)
The value and meaning of configuration parameters are discussed in the [comment.txt](config/comment.txt)  

Provide values for all parameters in configuration file.

## Steps to follow to run the project

1. Make sure all the required applications and packages are installed.
2. Make sure all the configuration parameters are provided.
3. Create the following folders before running the program,
   1. logs
   2. models
      1. model1
      2. model2
      3. model_pool
   3. output
4. Run the file [insert_imdb_dbpedia.py](utils/insert_imdb_dbpedia.py) which inserts entity descriptions from pool A, pool B, and ground truth labels into the mongoDB databases. It uses the database details provided in the config files.
5. Insert {"_id": 1, "flag": false} into MongoDB. For name of database and collection please refer to [model_deploy_util.py](utils/model_deploy_util.py).
6. Run the file [execute.py](kafka_input/execute.py) from [kafka_input](kafka_input) folder which starts The Prediction Layer, here sending the entity pairs. As input argument provide "producer".
7. Run the file [execute.py](kafka_input/execute.py) from [kafka_input](kafka_input) folder which starts The Prediction Layer, here classifying incoming entity pairs. As input argument provide "consumer".
8. Run the file [execute.py](pools/execute.py) from [pools][pools] folder which starts The Training Layer.
9. All the logs are stored in the [logs](logs) folder and the model metrics related data are stored in the [output](output) folder.


## The Team
Incremental Contrastive Learning for Entity Resolution on Dynamic Data devised by University of Stuttgart grad student [Maksim Helmann](https://de.linkedin.com/in/maksim-helmann-60b8701b1), under the supervision of [Prof. Dr. rer. nat. Melanie Herschel](https://www.f05.uni-stuttgart.de/fakultaet/personen/Herschel/). Intial framework was developed by University of Stuttgart grad student [Suhas Devendrakeerti Sangolli](www.linkedin.com/in/suhas2910). 
