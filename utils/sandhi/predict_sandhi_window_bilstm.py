import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import faulthandler

# Enable the fault handler on all threads
faulthandler.enable(all_threads=True)



def train_predict_sandhi_window(dtrain, dtest, mode):
    # --- MEMORY & BATCH SETTINGS ---
    MAX_LEN = 100         # truncate/pad to 100 chars
    TRAIN_BATCH = 128      # batch size for training
    INF_BATCH = 512        # batch size for inference
    epochs = 40
    latent_dim = 64
    inwordlen = 5

    # --- ENABLE GPU MEMORY GROWTH ---
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # --- VECTORIZE TRAINING DATA ---
    inputs, targets, chars = [], [], set()
    for data in dtrain:
        w = data[3]
        tgt = np.zeros(len(w), dtype=np.float32)
        for i in range(data[4], data[5]):
            tgt[i] = 1.0
        inputs.append(w)
        targets.append(tgt)
        chars.update(w)

    # clamp sequence length
    true_len = max(len(w) for w in inputs)
    maxlen = min(true_len, MAX_LEN)
    print(f"Using maxlen={maxlen} (clamped from {true_len})")

    chars |= {'*', '\\'}
    char2idx = {c: i for i, c in enumerate(sorted(chars))}
    num_tokens = len(chars)

    # --- BUILD X_train & Y_train ---
    X_ids = [[char2idx[c] for c in w[:maxlen]] for w in inputs]
    X_pad = pad_sequences(
        X_ids,
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=char2idx['*']
    )
    X_train = tf.one_hot(X_pad, depth=num_tokens, dtype=tf.float32)

    Y_pad = pad_sequences(
        targets,
        maxlen=maxlen,
        padding='post',
        truncating='post',
        value=0.0
    )
    Y_train = Y_pad.reshape(-1, maxlen, 1).astype(np.float32)

    print('Train samples:', len(X_train), 'Vocab size:', num_tokens)

    # --- MODEL DEFINITION ---
    encoder_in = Input(shape=(maxlen, num_tokens))
    bilstm = Bidirectional(
        LSTM(latent_dim, return_sequences=True, return_state=True)
    )
    out, fh, fc, bh, bc = bilstm(encoder_in)
    outd = Dropout(0.5)(out)
    decoder_out = Dense(1, activation='sigmoid')(outd)
    model = Model(encoder_in, decoder_out)
    model.compile(
        optimizer='rmsprop',
        loss='mean_squared_error',
        metrics=['accuracy']
    )
    model.summary()

    # --- TRAIN ---
    model.fit(
        X_train, Y_train,
        batch_size=TRAIN_BATCH,
        epochs=epochs,
        validation_split=0.1
    )

    # --- BATCHED INFERENCE ---
    total = len(dtest)
    print(f"Running inference on {total} samples in batches of {INF_BATCH}")
    passed = failed = 0
    startlist = []
    # iterate batches over test data
    for start_idx in range(0, total, INF_BATCH):
        end_idx = min(start_idx + INF_BATCH, total)
        print(f"  Predicting batch {start_idx}-{end_idx-1} of {total}")
        batch = dtest[start_idx:end_idx]
        bsize = len(batch)
        # build batch input
        X_batch = np.zeros((bsize, maxlen, num_tokens), dtype=np.float32)
        act_batch = np.zeros((bsize, maxlen), dtype=np.float32)
        for j, data in enumerate(batch):
            w = data[3]
            # encode
            for t, ch in enumerate(w[:maxlen]):
                X_batch[j, t, char2idx.get(ch, char2idx['*'])] = 1.0
            if len(w) < maxlen:
                X_batch[j, len(w):, char2idx['*']] = 1.0
            # actual target array
            for pos in range(data[4], data[5]):
                if pos < maxlen:
                    act_batch[j, pos] = 1.0
        # model prediction
        res = model.predict(X_batch, verbose=0)
        res = np.squeeze(res, axis=-1)  # shape (bsize, maxlen)

        # sliding-window and count
        for j in range(bsize):
            scores = res[j]
            actual = act_batch[j]
            best_sum = -1.0
            best_idx = 0
            for k in range(maxlen - inwordlen + 1):
                window_sum = np.sum(scores[k:k+inwordlen])
                if window_sum > best_sum:
                    best_sum = window_sum
                    best_idx = k
            startlist.append(best_idx)
            if mode == 0:
                true_positions = np.where(actual == 1)[0]
                first_true = int(true_positions[0]) if true_positions.size else -1
                if first_true == best_idx:
                    passed += 1
                else:
                    failed += 1
    if mode == 0:
        print('Passed:', passed, 'Failed:', failed)
    with open("output/startlist.pkl", "wb") as f:
        pickle.dump(startlist, f)
    print("Startlist saved to startlist.pkl")
    return startlist
