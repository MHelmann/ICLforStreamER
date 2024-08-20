import sys
import json
import random


def get_label(ground_truth, _id_a, _id_b):
    """
    This function finds the ground truth label for given entity pairs.
    :param ground_truth_col: Ground-truth collection name.
    :param _id_a: Entity-a id.
    :param _id_b: Entity-b id.
    :return : Ground truth label of entity pairs.
    """
    _id_a = int(_id_a)
    _id_b = int(_id_b)
    
    # Define the pair of entities you want to check
    entity_pair_to_check = {"entity_a": _id_a, "entity_b": _id_b}

    # Check if the pair of entities exists in the list of entries
    try:
        pair_exists = entity_pair_to_check in ground_truth
    except Exception:
        print("The pair of entities does not exist in the JSON file.")
        raise
    else:
        if pair_exists:
            return 1
        else:
            return 0

def label_datasets(original_file, ground_truth):
    """label dataset from input"""

    with open(original_file, 'r') as f:
        lines = f.readlines()
    f.close()

        # Read JSON file
    with open(ground_truth, 'r') as file:
        json_data = json.load(file)
    file.close()

    labeled_1 = []
    labeled_0 = []

    for i, line in enumerate(lines):
        line = line.strip()
        line = line.split(",")
        _id_a = int(line[0])
        _id_b = int(line[1])
        weight = line[2]
        label = get_label(json_data, _id_a, _id_b)    
        result = ','.join([str(_id_a), str(_id_b), weight, str(label)]) + '\n'
        if label == 1:
            labeled_1.append(result)
        else:
            labeled_0.append(result)

        if (i+1) % 500000 == 0:
            print(i)

    new_1 = 'input/Movies_input_1.txt'
    with open(new_1, 'w') as new1:
        new1.writelines(labeled_1)
    new1.close()
    new_0 = 'input/Movies_input_0.txt'
    with open(new_0, 'w') as new0:
        new0.writelines(labeled_0)
    new0.close()


def check_matches(file):
    with open(file, 'r') as f:
        lines = f.readlines()
    f.close()

    i = 0
    for _, line in enumerate(lines):
        line = line.strip()
        line = line.split(",")
        if line[3] == str(1):
            i += 1

    print(i)


def generate_skewed_dataset(label_1_set, label_0_set, skewness):
    # 0: 6004862
    # 1: 21596
    with open(label_1_set, 'r') as f1:
        lines_1 = f1.readlines()
    f1.close()

    with open(label_0_set, 'r') as f0:
        lines_0 = f0.readlines()
    f0.close()

    random.shuffle(lines_1)
    random.shuffle(lines_0)

    if skewness == 0.5:
        total_set_size = 2 * len(lines_1)
        data = []
        for l1 in lines_1:
            l1 = l1.replace(',1\n', '\n')
            data.append(l1)
        for l0 in lines_0[:len(lines_1)]:
            l0 = l0.replace(',0\n', '\n')
            data.append(l0)
        random.shuffle(data)
        assert len(data) == total_set_size, f"Expected total set size of {total_set_size}, but found {len(data)}"
        new_file50 = 'input/Movies_50perc.txt'
        with open(new_file50, 'w') as f50:
            f50.writelines(data)
        f50.close()
    elif skewness == 0.2:
        non_m_size = 4 * len(lines_1)
        print(non_m_size)
        total_set_size = len(lines_1) + non_m_size
        data = []
        for l1 in lines_1:
            l1 = l1.replace(',1\n', '\n')
            data.append(l1)
        for l0 in lines_0[:non_m_size]:
            l0 = l0.replace(',0\n', '\n')
            data.append(l0)
        random.shuffle(data)
        assert len(data) == total_set_size, f"Expected total set size of {total_set_size}, but found {len(data)}"
        new_file20 = 'input/Movies_20perc.txt'
        with open(new_file20, 'w') as f20:
            f20.writelines(data)
        f20.close()
    elif skewness == 0.1:
        non_m_size = 9 * len(lines_1)
        print(non_m_size)
        total_set_size = len(lines_1) + non_m_size
        data = []
        for l1 in lines_1:
            l1 = l1.replace(',1\n', '\n')
            data.append(l1)
        for l0 in lines_0[:non_m_size]:
            l0 = l0.replace(',0\n', '\n')
            data.append(l0)
        random.shuffle(data)
        assert len(data) == total_set_size, f"Expected total set size of {total_set_size}, but found {len(data)}"
        new_file10 = 'input/Movies_10perc.txt'
        with open(new_file10, 'w') as f10:
            f10.writelines(data)
        f10.close()
    elif skewness == 0.05:
        non_m_size = 19 * len(lines_1)
        print(non_m_size)
        total_set_size = len(lines_1) + non_m_size
        data = []
        for l1 in lines_1:
            l1 = l1.replace(',1\n', '\n')
            data.append(l1)
        for l0 in lines_0[:non_m_size]:
            l0 = l0.replace(',0\n', '\n')
            data.append(l0)
        random.shuffle(data)
        assert len(data) == total_set_size, f"Expected total set size of {total_set_size}, but found {len(data)}"
        new_file5 = 'input/Movies_5perc.txt'
        with open(new_file5, 'w') as f5:
            f5.writelines(data)
        f5.close()



if __name__ == "__main__":
    ground_truth = '/Users/helmann/research/input/entity_pool/movies/ground_truth_moviesIdDuplicates.json'
    original_file = 'input/Movies_input.txt'
    #label_datasets(original_file, ground_truth)
    #check_matches('/Users/helmann/research/input/Movies_input_1.txt')

    label_1 = 'input/Movies_input_1.txt'
    label_0 = 'input/Movies_input_0.txt'
    #generate_skewed_dataset(label_1, label_0, 0.05)




    ''''
    Everything is done for movie dataset like in Master thesis because the feature names for the Movies dataset, 
    however, differ according to the sources. As a result, in the majority of the experiments, we utilized the 
    Movies dataset to assess our model.
    x label dataset from input (1x function call for each dataset [3 in tota] - function A)
    x load in + separate according to match/non-match (function B)
    x shuffle (for data skewness - function B)
    x define input dataset size (for data skewness - function B)
    x draw x% from match subset and 1-x% from non-match (for data skewness - function B)
    x combine subsets --> final input dataset (for data skewness - function B)
    - create batches to send later during training
    '''
