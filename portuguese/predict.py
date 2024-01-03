# 1) get the output of read_data
# 2) predict according to the English co-reference model
# 3) compare the output of step (2) with the golden labels of step (1)
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from transformers import AutoTokenizer
import argparse

from read_data import read_conll


def main():

    parser = argparse.ArgumentParser(description='Prediction of coreference for a CONLL file.')
    parser.add_argument('--input', required=True, help='Path to the input CoNLL file')

    args = parser.parse_args()
    file_path = args.input

    mentions_clusters, sentences = read_conll(file_path)

    model_name = 'VincentNLP/seq2seq-coref-t0-3b-partial-linear'

    config = AutoConfig.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)

    for doc in sentences:
    	for idx_sent, sent in enumerate(sentences[doc]):
            input_ids = tokenizer(sent, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, max_length=60)
            clusters = tokenizer.batch_decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
	main()