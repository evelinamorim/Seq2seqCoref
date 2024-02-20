import argparse
import re
from nltk import word_tokenize
import random

punctuation = ".!?:;,"

# Set the random seed for reproducibility
random.seed(42)  # You can use any integer value here

def read_conll(file_path):
    mentions_clusters = {}
    sentences = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        current_document = None
        current_clusters = {}
        open_cluster = set()

        sent_id = 0
        sent_txt = ""
        for line in file:
            if line.startswith("#begin document"):
                current_document = re.search(r'#begin document (.+\.txt\.xml)', line).group(1)
                mentions_clusters[current_document] = []
                sentences[current_document] = []
                sent_id = 0
                sent_txt = ""
                current_clusters[current_document] = set()
            elif line.startswith("#end document"):
                if len(sent_txt) > 0:
                    sentences[current_document].append(sent_txt.rstrip())   

                current_document = None
            else:
                parts = line.strip().split('\t')
                if len(parts) > 1:

                    token_id = int(parts[0])
                    mention = parts[1]
                    clusters_str = parts[-1]


                    if token_id !=0 and sent_id!=0: 
                        if mention not in punctuation:
                            sent_txt = sent_txt + " " + mention
                        else:
                            sent_txt = sent_txt + mention + " "
                    else:
                        # first token in the sentence
                        if len(sent_txt) > 0:
                            sentences[current_document].append(sent_txt.rstrip()) 
                            #print(sentences[current_document])
                        sent_txt = mention
                        sent_id = sent_id + 1 


                    # Extract clusters from various formats
                    cluster_starts = re.findall(r'\((\d+)', clusters_str)
                    cluster_ends = re.findall(r'(\d+)\)', clusters_str)

                    if cluster_starts != [] or cluster_ends != []:

                        # to keep track if there is an open cluster
                        if cluster_starts != []:
                            open_cluster = open_cluster.union(set(cluster_starts))

                        if cluster_ends != []:
                            # so there is only closing cluster
                            # check if the opening cluster is closing
                            open_cluster.difference_update(set(cluster_ends))

                        # the union of all clusters that this mention belongs
                        clusters_ids = open_cluster.copy()
                        clusters_ids.update(set(cluster_starts))
                        clusters_ids.update(set(cluster_ends))
                        # TODO: offset ou (#sentence,#token)
                        mentions_clusters[current_document].append((mention,token_id, sent_id, clusters_ids))
                    else:
                        if open_cluster: # check if this token is in a cluster
                            mentions_clusters[current_document].append((mention,token_id, sent_id, open_cluster.copy()))


    return mentions_clusters, sentences

def build_splits(bin1, bin2):
    # Define the proportions for training, testing, and validation sets
    train_ratio = 0.7
    test_ratio = 0.2
    val_ratio = 0.1

    # Shuffle the documents in each bin to ensure randomness
    random.shuffle(bin1)
    random.shuffle(bin2)

    # Calculate the number of documents for each set based on the ratios
    num_bin1 = len(bin1)
    num_bin2 = len(bin2)

    num_train_bin1 = int(train_ratio * num_bin1)
    num_test_bin1 = int(test_ratio * num_bin1)
    num_val_bin1 = num_bin1 - num_train_bin1 - num_test_bin1

    num_train_bin2 = int(train_ratio * num_bin2)
    num_test_bin2 = int(test_ratio * num_bin2)
    num_val_bin2 = num_bin2 - num_train_bin2 - num_test_bin2

    # Split bin 1 into training, testing, and validation sets
    train_bin1 = bin1[:num_train_bin1]
    test_bin1 = bin1[num_train_bin1:num_train_bin1 + num_test_bin1]
    val_bin1 = bin1[num_train_bin1 + num_test_bin1:]

    # Split bin 2 into training, testing, and validation sets
    train_bin2 = bin2[:num_train_bin2]
    test_bin2 = bin2[num_train_bin2:num_train_bin2 + num_test_bin2]
    val_bin2 = bin2[num_train_bin2 + num_test_bin2:]

    return train_bin1 + train_bin2, test_bin1 + test_bin1, val_bin1 + val_bin2


def main():
    parser = argparse.ArgumentParser(description='Extract mentions and clusters from CoNLL file.')
    parser.add_argument('--input', required=True, help='Path to the input CoNLL file')

    args = parser.parse_args()
    file_path = args.input

    mentions_clusters, sentences = read_conll(file_path)
    ntokens = []
    hist_ = {"<=500":0,">500_<=2000":0,">=2000":0}

    split = {"500":[],"2000":[]}
    for doc in sentences:
        count_tokens = 0
        for sent in sentences[doc]:
            count_tokens += len(word_tokenize(sent))
        if count_tokens <= 500:
            hist_["<=500"] +=  1
            split["500"].append(doc)
        elif count_tokens > 500 and count_tokens <= 2000:
            hist_[">500_<=2000"] +=  1
            split["2000"].append(doc)
        else:
            hist_[">=2000"] +=  1
        ntokens.append(count_tokens)
    print(f"There are {len(sentences)} documents.")
    print(f"The average number of tokens is {sum(ntokens)/len(ntokens)}.")
    print(hist_)

    train_split, test_split, val_split = build_splits(split["500"], split["2000"])

    print(f"Train split lenght:{len(train_split)}")
    print(f"Test split lenght:{len(test_split)}")
    print(f"Validation split lenght:{len(val_split)}")

    with open("train.split", "w") as fd:
        for file_name in train_split:
            fd.write(f"{file_name}\n")

    with open("test.split", "w") as fd:
        for file_name in test_split:
            fd.write(f"{file_name}\n")

    with open("val.split", "w") as fd:
        for file_name in val_split:
            fd.write(f"{file_name}\n")


    #print(len(sentences["D1_C30_Folha_07-08-2007_09h19.txt.xml"]))
    #for document, mentions in mentions_clusters.items():
    #    print(f'Documento: {document}')
    #    for mention, token_id, sent_id, clusters in mentions:
    #        print(f'Menção: {mention}, Token id:{token_id}, Sentence id:{sent_id} , Clusters: {clusters}')
    #        print(f'Sentence: {sentences[document][sent_id - 1]}')
    #    print()

if __name__ == "__main__":
    main()
