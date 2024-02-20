# 1) get the output of read_data
# 2) predict according to the English co-reference model
# 3) compare the output of step (2) with the golden labels of step (1)
import torch
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoTokenizer
import argparse
import time
import joblib

import os

from read_data import read_conll

def main():

    parser = argparse.ArgumentParser(description='Prediction of coreference for a CONLL file.')
    parser.add_argument('--input', required=True, help='Path to the input CoNLL file')

    args = parser.parse_args()
    file_path = args.input
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    mentions_clusters, sentences = read_conll(file_path)

    model_name = 'VincentNLP/seq2seq-coref-t0-3b-partial-linear'


    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name,model_max_length=4096)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

    model = model.to(device)

    num_parameters = sum(p.numel() for p in model.parameters())
    print("Number of Parameters:", num_parameters)
    output_path = "/home/uud15559/Seq2seqCoref/portuguese/output/"

    for doc in sentences:
        output_file = os.path.join(output_path, doc + ".joblib")
        if os.path.exists(output_file):
            continue

        start_time = time.time()

        print(f"Processing document {doc} {len(sentences[doc])}")
        text = " ".join(sentences[doc])

        input_ids = tokenizer(text, return_tensors="pt").input_ids
        if input_ids.shape[1] < 2048 or input_ids.shape[1] > 4096:
          continue
        print(">>>", input_ids.shape[1])
        input_ids = input_ids.to(device)
        outputs = model.generate(input_ids, max_length=4096)
        clusters = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        joblib.dump(clusters, output_file)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed Time:", elapsed_time, "seconds")



if __name__ == "__main__":
	main()
