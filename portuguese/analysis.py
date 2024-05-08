import joblib
import os

import re
import numpy as np

import unidecode

from sklearn.metrics import precision_score, recall_score, f1_score

from collections import defaultdict
from itertools import combinations

import argparse

import pdb

def build_clusters(data):
    clusters = defaultdict(list)
    for mention, cluster_number in data:
        clusters[cluster_number].append(mention)
    return clusters

def compute_muc(golden_data, pred_data):
    golden_clusters = build_clusters(golden_data)
    pred_clusters = build_clusters(pred_data)

    # Initialize variables to store mention-pair counts
    num_gold_pairs = 0
    num_pred_pairs = 0
    num_correct_pairs = 0

    # Count mention-pairs in golden clusters
    for cluster_mentions in golden_clusters.values():
        num_mentions = len(cluster_mentions)
        num_gold_pairs += num_mentions * (num_mentions - 1) // 2

    # Count mention-pairs in predicted clusters
    for cluster_mentions in pred_clusters.values():
        num_mentions = len(cluster_mentions)
        num_pred_pairs += num_mentions * (num_mentions - 1) // 2

    # Count correct mention-pairs (in same cluster in both golden and predicted)
    for gold_cluster_mentions in golden_clusters.values():
        for pred_cluster_mentions in pred_clusters.values():
            common_mentions = set(gold_cluster_mentions) & set(pred_cluster_mentions)
            num_common_pairs = len(list(combinations(common_mentions, 2)))

            num_correct_pairs += num_common_pairs

    # Calculate precision, recall, and F1 score
    precision = num_correct_pairs / num_pred_pairs if num_pred_pairs > 0 else 0
    recall = num_correct_pairs / num_gold_pairs if num_gold_pairs > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score

def compute_bcubed(golden_data, pred_data):
    golden_clusters = build_clusters(golden_data)
    pred_clusters = build_clusters(pred_data)

    # Initialize variables for precision and recall
    precision_sum = 0
    recall_sum = 0

    # Compute precision for each mention in predicted clusters
    for pred_cluster_mentions in pred_clusters.values():
        for mention in pred_cluster_mentions:
            # Find the corresponding golden cluster
            gold_cluster_number = next((cluster_number for cluster_number, mentions in golden_clusters.items() if mention in mentions), None)
            if gold_cluster_number is not None:
                # Count the number of correctly clustered mentions
                correct_mentions = sum(1 for m in pred_cluster_mentions if m in golden_clusters[gold_cluster_number])
                precision = correct_mentions / len(pred_cluster_mentions)
                precision_sum += precision

    # Compute recall for each mention in golden clusters
    for gold_cluster_mentions in golden_clusters.values():
        for mention in gold_cluster_mentions:
            # Find the corresponding predicted cluster
            pred_cluster_number = next((cluster_number for cluster_number, mentions in pred_clusters.items() if mention in mentions), None)
            if pred_cluster_number is not None:
                # Count the number of correctly clustered mentions
                correct_mentions = sum(1 for m in gold_cluster_mentions if m in pred_clusters[pred_cluster_number])
                recall = correct_mentions / len(gold_cluster_mentions)
                recall_sum += recall

    # Compute precision and recall averages
    num_pred_mentions = sum(len(cluster) for cluster in pred_clusters.values())
    num_gold_mentions = sum(len(cluster) for cluster in golden_clusters.values())

    precision_avg = precision_sum / num_pred_mentions if num_pred_mentions > 0 else 0
    recall_avg = recall_sum / num_gold_mentions if num_gold_mentions > 0 else 0

    # Compute F1 score
    f1_score = (2 * precision_avg * recall_avg) / (precision_avg + recall_avg) if precision_avg + recall_avg > 0 else 0

    return precision_avg, recall_avg, f1_score

def compute_ceaf(golden_data, pred_data):
    golden_clusters = build_clusters(golden_data)
    pred_clusters = build_clusters(pred_data)

    # Initialize variables for precision and recall
    num_correct_links = 0
    num_pred_links = 0
    num_gold_links = 0

    # Compute correct, predicted, and gold links
    for pred_cluster_mentions in pred_clusters.values():
        for mention1, mention2 in combinations(pred_cluster_mentions, 2):
            num_pred_links += 1
            if any(mention1 in cluster_mentions and mention2 in cluster_mentions for cluster_mentions in golden_clusters.values()):
                num_correct_links += 1

    for gold_cluster_mentions in golden_clusters.values():
        for mention1, mention2 in combinations(gold_cluster_mentions, 2):
            num_gold_links += 1

    # Calculate precision, recall, and F1 score
    precision = num_correct_links / num_pred_links if num_pred_links > 0 else 0
    recall = num_correct_links / num_gold_links if num_gold_links > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1_score


def join_tokens(tok_lst):

    special_suffix = ["se", "me", "te", "lhe", "o",
                      "a", "la", "lo", "lho", "lha", "nos", "feira"]
    parentheses = ["(", "{"]
    punctuation = [",", ".", ":", ";", "?", "!", ")", "}"]

    # first, find all hyphen in the token list
    array_tok_lst = np.array(tok_lst)
    indices = np.where(array_tok_lst == "-")[0]
    indices = list(indices)

    if indices != []:
        tok_join = ""
        if len(tok_lst) > 0:
            i = 0
            while i < len(tok_lst):
                if i in indices:
                    if i + 1 < len(tok_lst) and \
                            tok_lst[i + 1].lower() in special_suffix:
                        tok_join = tok_join.strip() + tok_lst[i]
                        i = i + 1

                if tok_lst[i] in parentheses:
                    tok_join = tok_join + tok_lst[i]
                else:
                    if tok_lst[i] in punctuation:
                        tok_join = tok_join.rstrip() + tok_lst[i] + " "
                    else:
                        tok_join = tok_join + tok_lst[i] + " "
                i = i + 1

        return tok_join.strip()
    else:
        tok_join = ""
        if len(tok_lst) > 0:
            i = 0
            while i < len(tok_lst):

                if tok_lst[i] in parentheses:
                    tok_join = tok_join + tok_lst[i]
                else:
                    if tok_lst[i] in punctuation:
                        tok_join = tok_join.rstrip() + tok_lst[i] + " "
                    else:
                        tok_join = tok_join + tok_lst[i] + " "
                i = i + 1

        return tok_join.strip()

def get_mentions_notag(mentions_output):
    """
    The input is a string of mentions in a not tag format

    the output is a list of tuples. Each tuple is a mention of an entity and 
    the cluster number
    """
    ans = []
    mentions = mentions_output.split("|")
    for m in mentions:
        match = re.search(r'\b\d+\b', m)
        #print(match, m)
        if match:
            number = int(match.group())
            start_span, end_span = match.span()
            mention_text = m[3+1:]
            
            ans.append((mention_text.strip(),number))
        #else:
        #    print(f"Warning: Cluster with an unique mention {m}")
        
    return ans


def get_mentions(mentions_output):
    """
    The input is a string of mentions in a tag format

    the output is a list of tuples. Each tuple is a mention of an entity and 
    the cluster number
    """
    ans = []
    mentions = re.findall("<m>[\s|\w|-]+<\/m>", mentions_output)
    for m in mentions:
        # ignoring the starting and ending tags <m> </m>
        mention_info = m[4:-4].split("|")

        if len(mention_info) > 1:
            mention_text = mention_info[0].strip()
            mention_cluster = int(mention_info[1])
            ans.append((mention_text, mention_cluster))
        else:
            print(f"Incomplete mention {m}")

    return ans


def read_conll(data_file):

    data_golden = {}
    stack_cluster = []
    stack_starts = []
    cluster_token_lst = []
    file_name = ""
    mention_lst = []

    with open(data_file, "r") as fd:
        for line in fd:
            elements = line.split()
            if len(elements) > 0:
                if elements[0].startswith("#begin"):
                    file_name = elements[-1].rstrip()
                    # data_golden[file_name] = []
                    stack_cluster = []
                    stack_starts = []
                    mention_lst = []
                    cluster_token_lst = []
                elif elements[0].startswith("#end"):

                    # just ignore end and blank lines

                    if mention_lst != []:
                        data_golden[file_name] = mention_lst
                else:
                    # these are data lines

                    # check if starts a mention cluster
                    m_starts = re.findall("\(\d+", elements[-1].rstrip())

                    # check if ends a group
                    m_ends = re.findall("\d+\)", elements[-1].rstrip())

                    if m_starts or m_ends or stack_cluster != []:
                        cluster_token_lst.append(elements[1])

                    for s in m_starts:
                        # it starts a cluster
                        cluster_number = int(s[1:])
                        stack_cluster.append(cluster_number)
                        stack_starts.append(len(cluster_token_lst) - 1)

                    for e in m_ends:
                        cluster_number_ends = int(e[:-1])
                        idx_cluster = stack_cluster.index(cluster_number_ends)
                        start_token = stack_starts[idx_cluster]

                        #cluster_txt = unidecode.unidecode(join_tokens(
                        #    cluster_token_lst[start_token:]))
                        cluster_txt = join_tokens(cluster_token_lst[start_token:])


                        stack_cluster.pop(idx_cluster)
                        stack_starts.pop(idx_cluster)

                        mention_lst.append((cluster_txt, cluster_number_ends))
    return data_golden


def compute_mentions_metrics(gold_mentions, pred_mentions):
    # Extract mention texts and cluster labels from gold mentions
    gold_mention_texts, gold_clusters = zip(*gold_mentions)

    # Extract mention texts and cluster labels from predicted mentions
    pred_mention_texts, pred_clusters = zip(*pred_mentions)

    # Calculate true positives, false positives, and false negatives
    true_positives = sum(
        1 for mention_text, cluster in gold_mentions if mention_text in pred_mention_texts and cluster in pred_clusters)
    false_positives = sum(
        1 for mention_text, cluster in pred_mentions if mention_text not in gold_mention_texts)
    false_negatives = sum(
        1 for mention_text, cluster in gold_mentions if mention_text not in pred_mention_texts)

    # Compute precision, recall, and F1 score
    precision = true_positives / \
        (true_positives + false_positives) if (true_positives +
                                               false_positives) > 0 else 0
    recall = true_positives / \
        (true_positives + false_negatives) if (true_positives +
                                               false_negatives) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# output_dir = "results_pt_mt5/"
# golden_data = "results/Corref-PT-SemEval_v2.v4_gold_conll"

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Read input and golden values")

    # Add arguments
    parser.add_argument("--input", "-i", type=str, help="Input value")
    parser.add_argument("--golden", "-g", type=str, help="Golden value")
    parser.add_argument("--cluster", "-c", default="notag",type=str, help="Cluster type (tag or notag)")


    # Parse the command line arguments
    args = parser.parse_args()

    # Access the input and golden values
    pred_data_dir = args.input
    golden_data = args.golden

    
    golden_mentions = read_conll(golden_data)
    predict_mentions = {}

    file_lst = os.listdir(pred_data_dir)

    for f in file_lst:
        if f.endswith(".joblib"):
            pred_file = os.path.join(pred_data_dir, f)
            data = joblib.load(pred_file)

            if args.cluster == "notag":
                mention_lst = get_mentions_notag(data[0])
            else:
                mention_lst = get_mentions(data[0])
            doc = os.path.splitext(f)[0]
            predict_mentions[doc] = mention_lst

    # entity identification
    # precision, recall, f1
    #0.4048749066550904 0.193332087449251 0.2418257674778436
 
    # TODO: investigar quando f1 eh 0?
    precision_final = 0
    recall_final = 0
    f1_final = 0

    p_muc_final = 0
    r_muc_final = 0
    f1_muc_final = 0

    p_b3_final = 0
    r_b3_final = 0
    f1_b3_final = 0

    p_ceaf_final = 0
    r_ceaf_final = 0
    f1_ceaf_final = 0


    for doc in golden_mentions:
        if doc in predict_mentions and len(predict_mentions[doc]) > 0:
            p_muc, r_muc, f1_muc = compute_muc(golden_mentions[doc], predict_mentions[doc])
            p_muc_final += p_muc
            r_muc_final += r_muc
            f1_muc_final += f1_muc

            p_b3, r_b3, f1_b3 = compute_bcubed(golden_mentions[doc], predict_mentions[doc])
            p_b3_final += p_b3
            r_b3_final += r_b3
            f1_b3_final += f1_b3

            p_ceaf, r_ceaf, f1_ceaf = compute_ceaf(golden_mentions[doc], predict_mentions[doc])
            p_ceaf_final += p_ceaf
            r_ceaf_final += r_ceaf
            f1_ceaf_final += f1_ceaf

            precision, recall, f1 = compute_mentions_metrics(
                golden_mentions[doc], predict_mentions[doc])
            precision_final += precision
            recall_final += recall
            f1_final += f1

    precision_final = precision_final / len(golden_mentions)
    recall_final = recall_final / len(golden_mentions)
    f1_final = f1_final / len(golden_mentions)

    print(precision_final, recall_final, f1_final)

    print("MUC")

    p_muc_final = p_muc_final / len(golden_mentions)
    r_muc_final = r_muc_final / len(golden_mentions)
    f1_muc_final = f1_muc_final / len(golden_mentions)

    print(p_muc_final, r_muc_final, f1_muc_final)

    print("B3")

    p_b3_final = p_b3_final / len(golden_mentions)
    r_b3_final = r_b3_final / len(golden_mentions)
    f1_b3_final = f1_b3_final / len(golden_mentions)

    print(p_b3_final, r_b3_final, f1_b3_final)

    print("CEAF")

    p_ceaf_final = p_ceaf_final / len(golden_mentions)
    r_ceaf_final = r_ceaf_final / len(golden_mentions)
    f1_ceaf_final = f1_ceaf_final / len(golden_mentions)

    print(p_ceaf_final, r_ceaf_final, f1_ceaf_final)

if __name__ == "__main__":
    main()
