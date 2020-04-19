import transformers


MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
BERT_PATH = "F:/Project/Jigsaw_multi_toxic/input/bert_base_multilingual_uncased/"
MODEL_PATH = "F:/Project/Jigsaw_multi_toxic/input/bert_base_multilingual_uncased/model.bin"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)