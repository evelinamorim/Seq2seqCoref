from transformers import HfArgumentParser, set_seed
from transformers import AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, AutoTokenizer
from transformers.integrations import TensorBoardCallback
from accelerate import Accelerator

from arguments import DataArguments, ModelArguments, CorefTrainingArguments \
    as TrainingArguments
from constants import SPEAKER_START, SPEAKER_END, MENTION_START, MENTION_END, \
    COPY, CLUSTER_NEW, CLUSTERS, SENTENCE_START, SENTENCE_END, SPECIAL_IDS, \
    NON_INT_SPECIAL_IDS, MARK_SPECIAL_IDS, MENTION_END_NON_INT_SPECIAL_IDS, \
    MENTION_ENDS
from data import CorefDataset
from trainer import CorefTrainer
import os

if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.abspath("/content/drive/MyDrive/UPorto/coref-experiments/args.json"))

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    num_new_tokens = tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                                           MENTION_START, MENTION_END,
                                           COPY])
    num_new_tokens += tokenizer.add_tokens([SENTENCE_START, SENTENCE_END])
    # we  need to resize model token embeddings
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path, config=config)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    train_set = CorefDataset(tokenizer, data_args, training_args, 'train')
    tb_callback = TensorBoardCallback()

    accelerator = Accelerator()
    trainer = accelerator.prepare(CorefTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_set,
        #        eval_dataset=dev_set,
        data_collator=collator,
        callbacks=[tb_callback]
    ))
    trainer.train()