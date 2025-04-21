from transformers import EncoderDecoderModel, BertTokenizer

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")  # Multilingual BERT tokenizer
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-multilingual-cased",  # Encoder (BERT)
    "bert-base-uncased"             # Decoder (BERT)
)

# Set special tokens
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id