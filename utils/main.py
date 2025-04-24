import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import sacrebleu

# 1) Pick the Indic→English OPUS‐MT model
MODEL_ID = "Helsinki-NLP/opus-mt-inc-en"

# 2) (Optional) if you hit 401 errors, authenticate:
#    >> huggingface-cli login
#    or pass use_auth_token=True below

# 3) Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)  # loads SentencePiece vocabs
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to("cuda")

model.eval()

# 4) Set up metrics
chrf = sacrebleu.CHRF()
bleu = sacrebleu.BLEU()
ter  = sacrebleu.TER()

# 5) Prepare your test split (assuming a `datasets` Translation feature)
from datasets import load_from_disk
ds = load_from_disk("./datasets/improved_itihasa_hf")
src_lang, tgt_lang = ds["test"].features["translation"].languages  # say ("sa","en")

preds, refs = [], []
BATCH = 16

for i in tqdm(range(0, len(ds["test"]), BATCH), desc="Translating"):
    batch = ds["test"][i : i + BATCH]["translation"]
    # 6) **Prefix** each Sanskrit input with the language‐token the model expects:
    src_texts = [f">>san_Deva<< {ex[src_lang]}" for ex in batch]
    tgt_texts = [ex[tgt_lang] for ex in batch]

    enc = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=256,
            num_beams=1,   # greedy decode for speed
        )
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    preds.extend(decoded)
    refs .extend(tgt_texts)

# 7) Compute corpus scores
refs_for_bleu = [refs]
print("chrF:", chrf.corpus_score(preds, refs_for_bleu).score)
print("BLEU:", bleu  .corpus_score(preds, refs_for_bleu).score)
print("TER: ", ter   .corpus_score(preds, refs_for_bleu).score)

# 8) Token‐accuracy (optional)
matched = total = 0
for p, r in zip(preds, refs):
    ptoks, rtoks = p.split(), r.split()
    for j in range(max(len(ptoks), len(rtoks))):
        total += 1
        if j < len(ptoks) and j < len(rtoks) and ptoks[j] == rtoks[j]:
            matched += 1
print("TokAcc:", matched/total*100)
