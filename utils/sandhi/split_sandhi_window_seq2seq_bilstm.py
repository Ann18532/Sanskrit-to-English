import os
import pickle
import faulthandler

# Enable C‐level backtraces on segfault/abort
faulthandler.enable()

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    Concatenate, Dropout
)
from sklearn.model_selection import train_test_split
import train_test_data_prepare as sdp


def train_sandhi_split(dtrain, dtest, mode):
    # ─── Hyperparameters ─────────────────────────────────────────────
    TRAIN_BS   = 128    # training batch size
    INF_BS     = 512   # inference batch size
    EPOCHS     = 40     # training epochs
    LATENT_DIM = 128   # LSTM hidden size

    # ─── 1) Build char set from train+test ──────────────────────────
    input_texts, target_texts = [], []
    X_tests, Y_tests = [], []
    chars = set()
    for src, join, inp, *rest in dtrain:
        tgt = '&' + src + '+' + join + '$'
        input_texts.append(inp); target_texts.append(tgt)
        chars.update(inp); chars.update(tgt)
    for src, join, inp, *rest in dtest:
        tgt = '&' + src + '+' + join + '$'
        X_tests.append(inp); Y_tests.append(tgt)
        chars.update(inp); chars.update(tgt)
    chars.add('*')
    characters = sorted(chars)
    num_tokens = len(characters)
    token_index = {c:i for i,c in enumerate(characters)}
    reverse_target_char_index = {i:c for c,i in token_index.items()}

    max_enc = max(len(s) for s in input_texts)
    max_dec = max(len(s) for s in target_texts)

    print(f"Samples: {len(input_texts)}, Vocab: {num_tokens}, Enc_len: {max_enc}, Dec_len: {max_dec}")

    # ─── 2) One‐hot encode training data ─────────────────────────────
    N = len(input_texts)
    enc_in = np.zeros((N, max_enc, num_tokens), dtype='float32')
    dec_in = np.zeros((N, max_dec, num_tokens), dtype='float32')
    dec_tr = np.zeros((N, max_dec, num_tokens), dtype='float32')
    for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
        for t, ch in enumerate(inp):
            enc_in[i,t,token_index[ch]] = 1.
        enc_in[i,len(inp):,token_index['*']] = 1.
        for t, ch in enumerate(tgt):
            dec_in[i,t,token_index[ch]] = 1.
            if t>0:
                dec_tr[i,t-1,token_index[ch]] = 1.
        dec_in[i,len(tgt):,token_index['*']] = 1.
        dec_tr[i,len(tgt):,token_index['*']] = 1.

    # ─── 3) Build seq2seq model ────────────────────────────────────
    # Encoder
    enc_inputs = Input(shape=(None, num_tokens), name='enc_in')
    bi_enc = Bidirectional(
        LSTM(LATENT_DIM, return_state=True, dropout=0.5),
        name='encoder_bi'
    )
    _, fh, fc, bh, bc = bi_enc(enc_inputs)
    state_h = Concatenate()([fh, bh])
    state_c = Concatenate()([fc, bc])
    enc_states = [state_h, state_c]

    # Decoder
    dec_inputs = Input(shape=(None, num_tokens), name='dec_in')
    dec_lstm = LSTM(
        LATENT_DIM*2,
        return_sequences=True,
        return_state=True,
        dropout=0.5,
        name='decoder_lstm'
    )
    dec_seq, _, _ = dec_lstm(dec_inputs, initial_state=enc_states)
    dec_dense = Dense(num_tokens, activation='softmax', name='decoder_dense')
    dec_out = dec_dense(dec_seq)

    model = Model([enc_inputs, dec_inputs], dec_out, name='seq2seq')
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── 4) Train ─────────────────────────────────────────────────
    model.fit(
        [enc_in, dec_in], dec_tr,
        batch_size=TRAIN_BS,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=1
    )
    model.save('bis2s.h5')
    print("✔ Model trained and saved")

    # ─── 5) Build inference models ─────────────────────────────────
    encoder_model = Model(enc_inputs, enc_states, name='encoder_model')
    s_h = Input(shape=(LATENT_DIM*2,), name='s_h')
    s_c = Input(shape=(LATENT_DIM*2,), name='s_c')
    out2, h2, c2 = dec_lstm(dec_inputs, initial_state=[s_h, s_c])
    out2 = dec_dense(out2)
    decoder_model = Model(
        [dec_inputs, s_h, s_c],
        [out2, h2, c2],
        name='decoder_model'
    )
    print("✔ Inference models ready")

    # ─── 6) Batch decoding with tf.while_loop ──────────────────────
    @tf.function
    def decode_batch(h0, c0):
        batch_sz = tf.shape(h0)[0]
        eos_id   = tf.constant(token_index['$'], tf.int32)

        # finished flags
        finished = tf.zeros((batch_sz,), tf.bool)
        # initial input tokens = '&'
        next_id  = tf.fill((batch_sz,), token_index['&'])
        # TensorArray to collect outputs
        ta       = tf.TensorArray(tf.int32, size=max_dec)
        h, c     = h0, c0
        t        = tf.constant(0)

        def cond(t, finished, h, c, ta, next_id):
            return tf.logical_and(t < max_dec,
                                  tf.logical_not(tf.reduce_all(finished)))

        def body(t, finished, h, c, ta, next_id):
            # one-hot encode next_id
            inp = tf.one_hot(next_id, num_tokens)        # [B, num_tokens]
            inp = tf.expand_dims(inp, 1)                 # [B,1,num_tokens]
            out, h_new, c_new = decoder_model([inp, h, c], training=False)
            logits        = out[:,0,:]                   # [B, num_tokens]
            next_id_new   = tf.cast(tf.argmax(logits, -1), tf.int32)
            ta            = ta.write(t, next_id_new)
            # update finished
            finished_new  = tf.logical_or(finished,
                                          tf.equal(next_id_new, eos_id))
            return (t+1,
                    finished_new,
                    h_new,
                    c_new,
                    ta,
                    next_id_new)

        t_final, finished_final, h_final, c_final, ta_final, _ = tf.while_loop(
            cond, body,
            [t, finished, h, c, ta, next_id],
            maximum_iterations=max_dec
        )
        # stack and transpose to [B, T]
        tokens = ta_final.stack()             # [T, B]
        tokens = tf.transpose(tokens, [1,0]) # [B, T]
        return tokens

    # ─── 7) Infer in batches ────────────────────────────────────────
    total = len(X_tests)
    results = []
    print(f"▶ Inference on {total} samples in batches of {INF_BS}")
    for start in range(0, total, INF_BS):
        end = min(start+INF_BS, total)
        print(f"  • Encoding batch {start}-{end-1} of {total}")

        # build one-hot encoder input
        B = end - start
        Xb = np.zeros((B, max_enc, num_tokens), dtype='float32')
        for j, seq in enumerate(X_tests[start:end]):
            for t, ch in enumerate(seq):
                if t>=max_enc: break
                Xb[j,t,token_index[ch]] = 1.
            if len(seq)<max_enc:
                Xb[j,len(seq):,token_index['*']] = 1.

        # encode to states
        h_batch, c_batch = encoder_model.predict(Xb, verbose=0)
        # batch decode
        decoded_ids = decode_batch(tf.constant(h_batch), tf.constant(c_batch)).numpy()

        # convert each row of ids to string
        for row in decoded_ids:
            s = []
            for idx in row:
                if idx == token_index['$']:
                    break
                s.append(reverse_target_char_index[int(idx)])
            results.append(''.join(s))

        print(f"  ✓ Finished batch up to sample {end}/{total}")

    # ─── 8) Save results ────────────────────────────────────────────
    os.makedirs('output', exist_ok=True)
    with open('output/results.pkl','wb') as f:
        pickle.dump(results, f)
    print("✔ Results saved to output/results.pkl")

    return results
