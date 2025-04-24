import gc, torch
import os
import logging
import faulthandler
import warnings
warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`")

# Enable the fault handler on all threads
faulthandler.enable(all_threads=True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"




torch.backends.cudnn.benchmark = True
# Disable TensorRT and GPU usage
# os.environ["TF_TRT"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow memory growth
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s):")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No GPU detected, defaulting to CPU.")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from datasets import load_from_disk, DatasetDict, Dataset
import numpy as np
import sacrebleu
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    BitsAndBytesConfig
)
from datetime import datetime

from peft import LoraConfig, get_peft_model

ds = load_from_disk("./datasets/improved_itihasa_hf")


chrf = sacrebleu.CHRF()
bleu = sacrebleu.BLEU()
ter  = sacrebleu.TER()


def compute_metrics(preds, labels, tokenizer):
    # decode predictions & labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    refs = [[lbl] for lbl in decoded_labels]

    # sacrebleu scores
    return {
        "chrF":    chrf.corpus_score(decoded_preds, refs).score,
        "BLEU":    bleu.corpus_score(decoded_preds, refs).score,
        "TER":     ter.corpus_score(decoded_preds, refs).score,
        "Tok.Acc": (
            sum(p==r for pr, rl in zip(decoded_preds, decoded_labels)
                   for p, r in zip(pr.split(), rl.split()))
            / sum(len(rl.split()) for rl in decoded_labels)
            * 100.0
        ),
    }


# 3) FINETUNE & EVALUATE Byte-to-Byte T5 variants
models = {
    # "google/byt5-base":  "B2B-Base",
    "google/byt5-small": "B2B-Small",
    # "google/byt5-large": "B2B-Large",
}

# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",              # use NF4 quantization
#     bnb_4bit_compute_dtype=torch.float16,   # keep compute in FP16
# )

for model_name, tag in models.items():
    print(f"\n=== Training {tag} ({model_name}) ===")
    # torch.cuda.empty_cache()
    # gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        # quantization_config=quant_config,         # ← quantize weights
    device_map={'':0},
    )

    # lora_cfg = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     target_modules=["q","v"],        # ← use T5’s actual layer names
    #     inference_mode=False
    # )
    # model = get_peft_model(model, lora_cfg)



    src_lang, tgt_lang = ds["train"].features["translation"].languages    
    print("→ src_lang:", src_lang, " tgt_lang:", tgt_lang)

    # 2) update your preprocess function
    def preprocess(batch):
        # batch["translation"] is a list of dicts: [{"en": "...", "sa": "..."}, ...]
        src_texts = [ex[src_lang] for ex in batch["translation"]]
        tgt_texts = [ex[tgt_lang] for ex in batch["translation"]]

        model_inputs = tokenizer(
            src_texts,
            truncation=True,
            padding="longest",
            max_length=512,
        )
        labels = tokenizer(
            tgt_texts,
            truncation=True,
            padding="longest",
            max_length=512,
        ).input_ids

        # Make sure there are no out-of-bounds indices
        vocab_size = tokenizer.vocab_size
        for i in range(len(model_inputs["input_ids"])):
            model_inputs["input_ids"][i] = [
                idx if idx < vocab_size else tokenizer.unk_token_id 
                for idx in model_inputs["input_ids"][i]
            ]
        
        # Also check labels for out-of-bounds
        for i in range(len(labels)):
            labels[i] = [
                idx if idx < vocab_size else tokenizer.unk_token_id 
                for idx in labels[i]
            ]

        model_inputs["labels"] = labels
        return model_inputs

    # 3) map over the DatasetDict, removing the single 'translation' column
    tokenized = ds.map(
        preprocess,
        batched=True,
        batch_size=64,
        remove_columns=["translation"],
        desc="Tokenizing dataset",
    )
    data_coll = DataCollatorForSeq2Seq(
        tokenizer, 
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
    )

    

    print("here")
    # for i in range(3):
    #     ex = tokenized["train"][i]
    #     print(f"Example {i}")
    #     print(" input_ids:", ex["input_ids"][:20])
    #     print(" attention_mask:", ex["attention_mask"][:20])
    #     print(" labels:", ex["labels"][:20])
    #     print()


    args = Seq2SeqTrainingArguments(
        output_dir=f"checkpoints/{tag}",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        predict_with_generate=False,
        evaluation_strategy="no",
        save_total_limit=2,
        num_train_epochs=3,
        # fp16=True,
        disable_tqdm=False,
        logging_strategy="steps",
        logging_steps=1,  # Show logs every 10 steps
    )


    print("idhar")
    small_val = tokenized["validation"].select(range(500))
    small_train = tokenized["train"].select(range(1000))

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset= tokenized["validation"],
        data_collator=data_coll,
        tokenizer=tokenizer,
    )

    trainer.train()

    print(f"\n--- Evaluating {tag} on test ---")
    result = trainer.predict(tokenized["test"])
    # if using generate(), predictions shape is (num_examples, seq_len, vocab_size)
    preds = (
        result.predictions.argmax(-1)
        if result.predictions.ndim == 3
        else result.predictions
    )
    metrics = compute_metrics(preds, result.label_ids, tokenizer)

    print(f"\nResults for {tag}:")
    for metric, score in metrics.items():
        print(f"  {metric:8s}: {score:.2f}")

    model.cpu()
    del model, trainer, tokenized, data_coll, tokenizer
    gc.collect()
    torch.cuda.empty_cache()