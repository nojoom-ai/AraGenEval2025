from transformers import MBart50TokenizerFast

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "ar_AR"
tokenizer.tgt_lang = "ar_AR"

def tokenize(example):
    model_inputs = tokenizer(
        example["input_text"], max_length=256, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["target_text"], max_length=256, truncation=True, padding="max_length"
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
