import os
import tensorflow as tf
import faulthandler

# Enable the fault handler on all threads
faulthandler.enable(all_threads=True)


# Disable TensorRT and GPU usage
# os.environ["TF_TRT"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from train_test_data_prepare import get_xy_data
from predict_sandhi_window_bilstm import train_predict_sandhi_window
from split_sandhi_window_seq2seq_bilstm import train_sandhi_split
from sklearn.model_selection import train_test_split
from datasets import load_from_disk, DatasetDict, Dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import devnagri_reader as dr
import sys
import pickle
# sys.path.append('/home/aryan/lbp/Sanskrit/Sanskrit-to-English')
from utils.dataset_processing import data_split as ds, datafetch as df

def slp1_to_devanagari(slp1_text):
    return transliterate(slp1_text, sanscript.SLP1, sanscript.DEVANAGARI)

swaras = ['a', 'A', 'i', 'I', 'u', 'U', 'e', 'E', 'o', 'O', 'f', 'F', 'x', 'X']
vyanjanas = ['k', 'K', 'g', 'G', 'N', 
             'c', 'C', 'j', 'J', 'Y',
             'w', 'W', 'q', 'Q', 'R',
             't', 'T', 'd', 'D', 'n',
             'p', 'P', 'b', 'B', 'm',
             'y', 'r', 'l', 'v','S', 'z', 's', 'h', 'L', '|']
others = ['H', 'Z', 'V', 'M', '~', '/', '\\', '^', '\'']

slp1charlist = swaras + vyanjanas + others

maxcompoundlen = 50
inwordlen = 5

def remove_nonslp1_chars(word):
    newword = ''
    for char in word:
        if char in slp1charlist:
            newword = newword + char
    return newword


inwordlen = 5
dataset = load_from_disk("./datasets/itihasa_dataset")
postprocess_dataset = DatasetDict()
print(dataset.shape)
print(dataset['train'][0])
print(dataset['train'][0]['translation'])

dtrain = get_xy_data("./datasets/Sandhikosh/sandhiset.txt")
dtest = ds.get_xy_data(dataset)

print(f"Training data size: {len(dtrain)}")
print(f"Testing data size: {len(dtest)}")

# Predict the sandhi window
# Define a more robust and configurable path for the pickle file
PICKLE_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # src/utils/sandhi
        "..", "..",                  # → src/
        "output",                    # → src/output
        "startlist.pkl"
    )
)

sl = []

if os.path.exists(PICKLE_PATH):
    # 1) Load the pre-computed startlist
    with open(PICKLE_PATH, "rb") as f:
        sl = pickle.load(f)
    print(f"Loaded startlist (n={len(sl)}) from {PICKLE_PATH}")

else:
    # 2) Otherwise, compute it (this runs your train_predict_sandhi_window
    #    or train_sandhi_split function which does training + prediction)
    sl = train_predict_sandhi_window(dtrain, dtest, mode=1)

print(len(sl))
if len(sl) == len(dtest):
    for i in range(len(dtest)):
        start = sl[i]
        end = sl[i] + inwordlen
        flen = len(dtest[i][3])
        if end > flen:
            end = flen
        dtest[i][2] = dtest[i][3][start:end]
        dtest[i][4] = start
        dtest[i][5] = end
else:
    print("error")

# Split the sandhi
results = train_sandhi_split(dtrain, dtest, 1)
sentences = {'train' : {}, 'validation' : {}, 'test' : {}}
# slpsentences = {'train' : {}, 'validation' : {}, 'test' : {}}



def _replace_sn_train(example, idx):
    # only replace when we have a new value for this index
    if idx in sentences['train']:
        # make sure translation exists and is a dict
        tr = example.get("translation", {})
        tr["sn"] = sentences['train'][idx]
        example["translation"] = tr
    return example

def _replace_sn_validation(example, idx):
    # only replace when we have a new value for this index
    if idx in sentences['validation']:
        # make sure translation exists and is a dict
        tr = example.get("translation", {})
        tr["sn"] = sentences['validation'][idx]
        example["translation"] = tr
    return example

def _replace_sn_test(example, idx):
    # only replace when we have a new value for this index
    if idx in sentences['test']:
        # make sure translation exists and is a dict
        tr = example.get("translation", {})
        tr["sn"] = sentences['test'][idx]
        example["translation"] = tr
    return example

if len(results) == len(dtest):
    passed = 0
    failed = 0
    print("Starting hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    for i in range(len(dtest)):
        if dtest[i][8] not in sentences[dtest[i][10]]:
            sentences[dtest[i][10]][dtest[i][8]] = ""
            # slpsentences[dtest[i][10]][dtest[i][8]] = ""
        if len(dtest[i][3]) <= 5 or dtest[i][9] == 0:
            # slpsentences[dtest[i][10]][dtest[i][8]] += " " + dtest[i][3]
            sans = slp1_to_devanagari(dtest[i][3])
            sentences[dtest[i][10]][dtest[i][8]] += " " + sans
            continue
        start = dtest[i][4]
        end = dtest[i][5]
        splitword = dtest[i][3][:start] + results[i] + dtest[i][3][end:]
        words = splitword.split('+')
        if len(words) != 2:
            # slpsentences[dtest[i][10]][dtest[i][8]] += " " + dtest[i][3]
            sans = slp1_to_devanagari(dtest[i][3])
            sentences[dtest[i][10]][dtest[i][8]] += " " + sans
            continue
        word1 = words[0].strip()
        word2 = words[1].strip()
        # slpsentences[dtest[i][10]][dtest[i][8]] += " " + word1 + " " + word2
        word1 = slp1_to_devanagari(word1)
        word2 = slp1_to_devanagari(word2)
        sentences[dtest[i][10]][dtest[i][8]] += " " + word1 + " " + word2
        actword = dtest[i][6] + '+' + dtest[i][7]
        if splitword == actword:
            passed = passed + 1
        else:
            failed = failed + 1
        
        # dataset['train'][dtest[i][8]]['translation']['sn'].map(sentences[dtest[i][8]])
   


train_mod = dataset["train"].map(
    _replace_sn_train,
    with_indices=True,
    # remove_columns=[]  # only needed if you’re dropping cols
)

validation_mod = dataset["validation"].map(
    _replace_sn_validation,
    with_indices=True,
    # remove_columns=[]  # only needed if you’re dropping cols
)
test_mod = dataset["test"].map(
    _replace_sn_test,
    with_indices=True,
    # remove_columns=[]  # only needed if you’re dropping cols
)


# 4. Reassemble your DatasetDict
updated = DatasetDict({
    "train":      train_mod,
    "validation": validation_mod,
    "test":       test_mod,
})

updated.save_to_disk("datasets/improved_itihasa_hf")
print("gg")

# for id in range(0, 2):
#     for wd in dataset['train'][id]['translation']['sn'].split(' '):
#         word1 = wd.strip()
#         word1 = dr.read_devnagri_text(word1)
#         slp1word1 = transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1)
#         slp1word1 = remove_nonslp1_chars(slp1word1)
#         print(slp1word1)
#     print("\n")


# print("new one\n")

# for id in range(0, 2):
#     for wd in updated['train'][id]['translation']['sn'].split(' '):
#         word1 = wd.strip()
#         word1 = dr.read_devnagri_text(word1)
#         slp1word1 = transliterate(word1, sanscript.DEVANAGARI, sanscript.SLP1)
#         slp1word1 = remove_nonslp1_chars(slp1word1)
#         print(slp1word1)
#     print("\n")