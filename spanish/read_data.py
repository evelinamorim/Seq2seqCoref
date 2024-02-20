import argparse
import re
from nltk import word_tokenize

punctuation = ".!?:;,"

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
                current_document = re.search(r'#begin document (.+\.tbf\.xml)', line).group(1)
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

def main():
    parser = argparse.ArgumentParser(description='Extract mentions and clusters from CoNLL file.')
    parser.add_argument('--input', required=True, help='Path to the input CoNLL file')

    args = parser.parse_args()
    file_path = args.input

    mentions_clusters, sentences = read_conll(file_path)
    ntokens = []
    for doc in sentences:
        count_tokens = 0
        for sent in sentences[doc]:
            count_tokens += len(word_tokenize(sent))
        ntokens.append(count_tokens)
    print(f"There are {len(sentences)} documents.")
    print(f"The average number of tokens is {sum(ntokens)/len(ntokens)}.")
    #print(len(sentences["D1_C30_Folha_07-08-2007_09h19.txt.xml"]))
    #for document, mentions in mentions_clusters.items():
    #    print(f'Documento: {document}')
    #    for mention, token_id, sent_id, clusters in mentions:
    #        print(f'Menção: {mention}, Token id:{token_id}, Sentence id:{sent_id} , Clusters: {clusters}')
    #        print(f'Sentence: {sentences[document][sent_id - 1]}')
    #    print()

if __name__ == "__main__":
    main()
