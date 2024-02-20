import joblib
import os

import re
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

def join_tokens(tok_lst):

    special_suffix = ["se","me", "te", "lhe", "o", "a", "la", "lo", "lho", "lha", "nos","feira"]
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

	with open(data_file,"r") as fd:
		for line in fd:
			elements = line.split()
			if len(elements) > 0:
			    if elements[0].startswith("#begin"):
				    file_name = elements[-1].rstrip()
				    #data_golden[file_name] = []
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
				    m_starts = re.findall("\(\d+",elements[-1].rstrip())

				    # check if ends a group
				    m_ends = re.findall("\d+\)",elements[-1].rstrip())

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
    true_positives = sum(1 for mention_text, cluster in gold_mentions if mention_text in pred_mention_texts and cluster in pred_clusters)
    false_positives = sum(1 for mention_text, cluster in pred_mentions if mention_text not in gold_mention_texts)
    false_negatives = sum(1 for mention_text, cluster in gold_mentions if mention_text not in pred_mention_texts)

    # Compute precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

output_dir = "portuguese/"
golden_data = "portuguese/Corref-PT-SemEval_v2.v4_gold_conll"

golden_mentions = read_conll(golden_data)
predict_mentions = {}

file_lst = os.listdir(output_dir)

for f in file_lst:
     if f.endswith(".joblib"):
     	output_file = os.path.join(output_dir, f)
     	data = joblib.load(output_file)

     	mention_lst = get_mentions(data[0])
     	doc = os.path.splitext(f)[0]
     	predict_mentions[doc] = mention_lst

# e se eu remover o acento dos textos?
#0.3090304705592262 0.15541242919758053 0.19135428278762848


# TODO: investigar quando f1 eh 0?
precision_final = 0
recall_final = 0
f1_final = 0
for doc in golden_mentions:
	if doc in predict_mentions:

		precision, recall, f1 = compute_mentions_metrics(golden_mentions[doc], predict_mentions[doc])
		precision_final += precision
		recall_final += recall
		f1_final += f1

precision_final = precision_final / len(golden_mentions)
recall_final = recall_final / len(golden_mentions)
f1_final = f1_final / len(golden_mentions)

print(precision_final, recall_final, f1_final)
