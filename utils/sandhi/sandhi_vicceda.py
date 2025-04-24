import gc, torch
import os
import logging
# import tensorflow as tf
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

# from train_test_data_prepare import get_xy_data
# from predict_sandhi_window_bilstm import train_predict_sandhi_window
# from split_sandhi_window_seq2seq_bilstm import train_sandhi_split
# from sklearn.model_selection import train_test_split
from datasets import load_from_disk, DatasetDict, Dataset
# from indic_transliteration import sanscript
# from indic_transliteration.sanscript import transliterate
# import devnagri_reader as dr
# import sys
# import pickle
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



# sys.path.append('/home/aryan/lbp/Sanskrit/Sanskrit-to-English')
# from utils.dataset_processing import data_split as ds, datafetch as df

# def slp1_to_devanagari(slp1_text):
#     return transliterate(slp1_text, sanscript.SLP1, sanscript.DEVANAGARI)

# swaras = ['a', 'A', 'i', 'I', 'u', 'U', 'e', 'E', 'o', 'O', 'f', 'F', 'x', 'X']
# vyanjanas = ['k', 'K', 'g', 'G', 'N', 
#              'c', 'C', 'j', 'J', 'Y',
#              'w', 'W', 'q', 'Q', 'R',
#              't', 'T', 'd', 'D', 'n',
#              'p', 'P', 'b', 'B', 'm',
#              'y', 'r', 'l', 'v','S', 'z', 's', 'h', 'L', '|']
# others = ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', '\'']

# slp1charlist = swaras + vyanjanas + others

# maxcompoundlen = 50
# inwordlen = 5

# def remove_nonslp1_chars(word):
#     newword = ''
#     for char in word:
#         if char in slp1charlist:
#             newword = newword + char
#     return newword


# # inwordlen = 5
# dataset = load_from_disk("./datasets/itihasa_dataset")
ds = load_from_disk("./datasets/improved_itihasa_hf")
# postprocess_dataset = DatasetDict()
# print(dataset.shape)
# print(dataset['train'][0])
# print(dataset['train'][0]['translation'])

# dtrain = get_xy_data("./datasets/Sandhikosh/sandhiset.txt")
# dtest = ds.get_xy_data(dataset)

# print(f"Training data size: {len(dtrain)}")
# print(f"Testing data size: {len(dtest)}")

# # Predict the sandhi window
# # Define a more robust and configurable path for the pickle file
# PICKLE_PATH = os.path.abspath(
#     os.path.join(
#         os.path.dirname(__file__),  # src/utils/sandhi
#         "..", "..",                  # → src/
#         "output",                    # → src/output
#         "startlist.pkl"
#     )
# )

# sl = []

# if os.path.exists(PICKLE_PATH):
#     # 1) Load the pre-computed startlist
#     with open(PICKLE_PATH, "rb") as f:
#         sl = pickle.load(f)
#     print(f"Loaded startlist (n={len(sl)}) from {PICKLE_PATH}")

# else:
#     # 2) Otherwise, compute it (this runs your train_predict_sandhi_window
#     #    or train_sandhi_split function which does training + prediction)
#     sl = train_predict_sandhi_window(dtrain, dtest, mode=1)

# print(len(sl))
# if len(sl) == len(dtest):
#     for i in range(len(dtest)):
#         start = sl[i]
#         end = sl[i] + inwordlen
#         flen = len(dtest[i][3])
#         if end > flen:
#             end = flen
#         dtest[i][2] = dtest[i][3][start:end]
#         dtest[i][4] = start
#         dtest[i][5] = end
# else:
#     print("error")

# # Split the sandhi
# results = train_sandhi_split(dtrain, dtest, 1)
# sentences = {'train' : {}, 'validation' : {}, 'test' : {}}
# # slpsentences = {'train' : {}, 'validation' : {}, 'test' : {}}



# def _replace_sn_train(example, idx):
#     # only replace when we have a new value for this index
#     if idx in sentences['train']:
#         # make sure translation exists and is a dict
#         tr = example.get("translation", {})
#         tr["sn"] = sentences['train'][idx]
#         example["translation"] = tr
#     return example

# def _replace_sn_validation(example, idx):
#     # only replace when we have a new value for this index
#     if idx in sentences['validation']:
#         # make sure translation exists and is a dict
#         tr = example.get("translation", {})
#         tr["sn"] = sentences['validation'][idx]
#         example["translation"] = tr
#     return example

# def _replace_sn_test(example, idx):
#     # only replace when we have a new value for this index
#     if idx in sentences['test']:
#         # make sure translation exists and is a dict
#         tr = example.get("translation", {})
#         tr["sn"] = sentences['test'][idx]
#         example["translation"] = tr
#     return example

# if len(results) == len(dtest):
#     passed = 0
#     failed = 0
#     print("Starting hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
#     for i in range(len(dtest)):
#         if dtest[i][8] not in sentences[dtest[i][10]]:
#             sentences[dtest[i][10]][dtest[i][8]] = ""
#             # slpsentences[dtest[i][10]][dtest[i][8]] = ""
#         if len(dtest[i][3]) <= 5 or dtest[i][9] == 0:
#             # slpsentences[dtest[i][10]][dtest[i][8]] += " " + dtest[i][3]
#             sans = slp1_to_devanagari(dtest[i][3])
#             sentences[dtest[i][10]][dtest[i][8]] += " " + sans
#             continue
#         start = dtest[i][4]
#         end = dtest[i][5]
#         splitword = dtest[i][3][:start] + results[i] + dtest[i][3][end:]
#         words = splitword.split('+')
#         if len(words) != 2:
#             # slpsentences[dtest[i][10]][dtest[i][8]] += " " + dtest[i][3]
#             sans = slp1_to_devanagari(dtest[i][3])
#             sentences[dtest[i][10]][dtest[i][8]] += " " + sans
#             continue
#         word1 = words[0].strip()
#         word2 = words[1].strip()
#         # slpsentences[dtest[i][10]][dtest[i][8]] += " " + word1 + " " + word2
#         word1 = slp1_to_devanagari(word1)
#         word2 = slp1_to_devanagari(word2)
#         sentences[dtest[i][10]][dtest[i][8]] += " " + word1 + " " + word2
#         actword = dtest[i][6] + '+' + dtest[i][7]
#         if splitword == actword:
#             passed = passed + 1
#         else:
#             failed = failed + 1
        
#         # dataset['train'][dtest[i][8]]['translation']['sn'].map(sentences[dtest[i][8]])
   


# train_mod = dataset["train"].map(
#     _replace_sn_train,
#     with_indices=True,
#     # remove_columns=[]  # only needed if you’re dropping cols
# )

# validation_mod = dataset["validation"].map(
#     _replace_sn_validation,
#     with_indices=True,
#     # remove_columns=[]  # only needed if you’re dropping cols
# )
# test_mod = dataset["test"].map(
#     _replace_sn_test,
#     with_indices=True,
#     # remove_columns=[]  # only needed if you’re dropping cols
# )


# # 4. Reassemble your DatasetDict
# updated = DatasetDict({
#     "train":      train_mod,
#     "validation": validation_mod,
#     "test":       test_mod,
# })

# updated.save_to_disk("datasets/improved_itihasa_hf")
# print("gg")

# for id in range(0, 2):
#     for wd in dataset['validation'][id]['translation']['sn'].split(' '):
#         word1 = wd.strip()
#         word1 = dr.read_devnagri_text(word1)
#         slp1word1 = transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1)
#         slp1word1 = remove_nonslp1_chars(slp1word1)
#         print(slp1word1)
#     print("\n")


# print("new one\n")

# for id in range(0, 2):
#     for wd in updated['validation'][id]['translation']['sn'].split(' '):
#         word1 = wd.strip()
#         word1 = dr.read_devnagri_text(word1)
#         slp1word1 = transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1)
#         slp1word1 = remove_nonslp1_chars(slp1word1)
#         print(slp1word1)
#     print("\n")


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
            # truncation=True,
            padding="longest",
            max_length=512,
        )
        labels = tokenizer(
            tgt_texts,
            # truncation=True,
            padding="longest",
            max_length=512,
        ).input_ids

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
        per_device_train_batch_size=4,
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
        train_dataset=small_train,
        eval_dataset= small_val,
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