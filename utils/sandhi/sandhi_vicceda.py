import os
import tensorflow as tf

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
import sys
# sys.path.append('/home/aryan/lbp/Sanskrit/Sanskrit-to-English')
from utils.dataset_processing import data_split as ds, datafetch as df

def slp1_to_devanagari(slp1_text):
    return transliterate(slp1_text, sanscript.SLP1, sanscript.DEVANAGARI)


inwordlen = 5
dataset = load_from_disk("./datasets/itihasa_dataset")
postprocess_dataset = DatasetDict()
print(dataset.shape)
print(dataset['train'][0]['translation'])

dtrain = get_xy_data("./datasets/Sandhikosh/sandhiset.txt")
dtest = ds.get_xy_data(dataset)

print(f"Training data size: {len(dtrain)}")
print(f"Testing data size: {len(dtest)}")

# Predict the sandhi window
sl = train_predict_sandhi_window(dtrain, dtest, 1)

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
sentences = {}
slpsentences = {}

if len(results) == len(dtest):
    passed = 0
    failed = 0
    print("Starting hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    for i in range(len(dtest)):
        if dtest[i][8] not in sentences:
            sentences[dtest[i][8]] = ""
            slpsentences[dtest[i][8]] = ""
        if len(dtest[i][3]) <= 5 or dtest[i][9] == 0:
            slpsentences[dtest[i][8]] += " " + dtest[i][3]
            sans = slp1_to_devanagari(dtest[i][3])
            sentences[dtest[i][8]] += " " + sans
            continue
        start = dtest[i][4]
        end = dtest[i][5]
        splitword = dtest[i][3][:start] + results[i] + dtest[i][3][end:]
        words = splitword.split('+')
        if len(words) != 2:
            slpsentences[dtest[i][8]] += " " + dtest[i][3]
            sans = slp1_to_devanagari(dtest[i][3])
            sentences[dtest[i][8]] += " " + sans
            continue
        word1 = words[0].strip()
        word2 = words[1].strip()
        slpsentences[dtest[i][8]] += " " + word1 + " " + word2
        word1 = slp1_to_devanagari(word1)
        word2 = slp1_to_devanagari(word2)
        sentences[dtest[i][8]] += " " + word1 + " " + word2
        actword = dtest[i][6] + '+' + dtest[i][7]
        print(splitword + "  "+ actword)
        if splitword == actword:
            passed = passed + 1
        else:
            failed = failed + 1
        
        dataset['train'][dtest[i][8]]['translation']['sn'].map(sentences[dtest[i][8]])
   
    print(slpsentences)
    print(sentences)
    
dataset.save_to_disk("./datasets/processed_itihasa_dataset")
dataset = load_from_disk("./datasets/processed_itihasa_dataset")
print(dataset['train'][0]['translation'])