import unittest


class MyImports(unittest.TestCase):
    def test_training_imports(self):
        from transformers import HfArgumentParser, set_seed
        from transformers import AutoModelForSeq2SeqLM, \
            DataCollatorForSeq2Seq, AutoConfig, AutoTokenizer
        from transformers.integrations import TensorBoardCallback

        from arguments import DataArguments, ModelArguments, CorefTrainingArguments \
            as TrainingArguments
        from constants import SPEAKER_START, SPEAKER_END, MENTION_START, MENTION_END, \
            COPY, CLUSTER_NEW, CLUSTERS, SENTENCE_START, SENTENCE_END, SPECIAL_IDS, \
            NON_INT_SPECIAL_IDS, MARK_SPECIAL_IDS, MENTION_END_NON_INT_SPECIAL_IDS, \
            MENTION_ENDS
        from data import CorefDataset, JointDataset
        from trainer import CorefTrainer
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
