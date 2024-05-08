import unittest

from transformers import T5Tokenizer

from data import CorefDataset
from arguments import DataArguments, CorefTrainingArguments as TrainingArguments


class TestDataReader(unittest.TestCase):
    def test_read_correfpt(self):
        int_tokenizer = T5Tokenizer.from_pretrained("google/mt5-small", model_max_length=4096)
        data_args = DataArguments()
        data_args.data_dir = "../data_jsonlines_pt"
        data_args.language = "portuguese"

        training_args = TrainingArguments(output_dir="../output_test")

        ds = CorefDataset(int_tokenizer, data_args, training_args, "train")
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
