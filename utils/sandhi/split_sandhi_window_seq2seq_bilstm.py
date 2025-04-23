import os
import pickle
import faulthandler

# Enable C‐level backtraces on segfault/abort
faulthandler.enable()

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Bidirectional,
    Concatenate, Dropout
)

def train_sandhi_split(dtrain, dtest, mode):
    # ─── Hyperparameters ─────────────────────────────────────────────
    batch_size = 128   # Batch size for training.
    epochs     = 1     # Number of epochs to train for.
    latent_dim = 128   # Latent dimensionality of the encoding space.
    INF_BS     = 512   # Batch size for inference.

    # ─── 1) Build character set from train+test ──────────────────────
    input_texts, target_texts = [], []
    X_tests, Y_tests = [], []
    chars = set()

    for src, join, inp, *rest in dtrain:
        tgt = '&' + src + '+' + join + '$'
        input_texts.append(inp)
        target_texts.append(tgt)
        chars.update(inp)
        chars.update(tgt)

    for src, join, inp, *rest in dtest:
        tgt = '&' + src + '+' + join + '$'
        X_tests.append(inp)
        Y_tests.append(tgt)
        chars.update(inp)
        chars.update(tgt)

    chars.add('*')
    characters = sorted(chars)
    num_tokens = len(characters)
    token_index = {c: i for i, c in enumerate(characters)}
    reverse_target_char_index = {i: c for c, i in token_index.items()}

    max_encoder_seq_length = max(len(s) for s in input_texts)
    max_decoder_seq_length = max(len(s) for s in target_texts)

    print('Number of samples:', len(input_texts))
    print('Number of unique tokens:', num_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    # ─── 2) One‐hot encode training data ─────────────────────────────
    N = len(input_texts)
    encoder_input_data = np.zeros((N, max_encoder_seq_length, num_tokens), dtype='float32')
    decoder_input_data = np.zeros((N, max_decoder_seq_length, num_tokens), dtype='float32')
    decoder_target_data = np.zeros((N, max_decoder_seq_length, num_tokens), dtype='float32')

    for i, (inp, tgt) in enumerate(zip(input_texts, target_texts)):
        # encoder
        for t, ch in enumerate(inp):
            encoder_input_data[i, t, token_index[ch]] = 1.
        encoder_input_data[i, len(inp):, token_index['*']] = 1.

        # decoder input & target (shifted)
        for t, ch in enumerate(tgt):
            decoder_input_data[i, t, token_index[ch]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, token_index[ch]] = 1.
        decoder_input_data[i, len(tgt):, token_index['*']] = 1.
        decoder_target_data[i, len(tgt):, token_index['*']] = 1.

    # ─── 3) Define seq2seq model ───────────────────────────────────
    encoder_inputs = Input(shape=(None, num_tokens), name='encoder_inputs')
    encoder = Bidirectional(
        LSTM(latent_dim, return_state=True, dropout=0.5),
        name='encoder_bi'
    )
    _, fh, fc, bh, bc = encoder(encoder_inputs)
    state_h = Concatenate()([fh, bh])
    state_c = Concatenate()([fc, bc])
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, num_tokens), name='decoder_inputs')
    decoder_lstm = LSTM(
        latent_dim * 2,
        return_sequences=True,
        return_state=True,
        dropout=0.5,
        name='decoder_lstm'
    )
    dec_out, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation='softmax', name='decoder_dense')
    decoder_outputs = decoder_dense(dec_out)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq')
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # ─── 4) Train ───────────────────────────────────────────────────
    model.fit(
        [encoder_input_data, decoder_input_data],
        decoder_target_data,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        verbose=1
    )
    model.save('bis2s.h5')
    print("✔ Trained model saved to bis2s.h5")

    # ─── 5) Build inference sub-models ──────────────────────────────
    encoder_model = Model(encoder_inputs, encoder_states, name='encoder_model')

    state_h_in = Input(shape=(latent_dim * 2,), name='state_h')
    state_c_in = Input(shape=(latent_dim * 2,), name='state_c')
    dec_out2, h2, c2 = decoder_lstm(decoder_inputs, initial_state=[state_h_in, state_c_in])
    dec_out2 = decoder_dense(dec_out2)
    decoder_model = Model(
        [decoder_inputs, state_h_in, state_c_in],
        [dec_out2, h2, c2],
        name='decoder_model'
    )
    print("✔ Sampling models ready")

    # ─── 6) Batched inference ───────────────────────────────────────
    total = len(X_tests)
    results = []
    print(f"▶ Inference on {total} samples in batches of {INF_BS}")
    for start in range(0, total, INF_BS):
        end = min(start + INF_BS, total)
        print(f"  • Encoding batch {start}-{end-1} of {total}")

        # one-hot encode this batch
        batch_size = end - start
        Xb = np.zeros((batch_size, max_encoder_seq_length, num_tokens), dtype='float32')
        for j, seq in enumerate(X_tests[start:end]):
            for t, ch in enumerate(seq):
                if t >= max_encoder_seq_length: break
                Xb[j, t, token_index[ch]] = 1.
            if len(seq) < max_encoder_seq_length:
                Xb[j, len(seq):, token_index['*']] = 1.

        # encode batch states
        h_batch, c_batch = encoder_model.predict(Xb, verbose=1)

        # decode each sample with EOS + max-length guard
        for h_arr, c_arr in zip(h_batch, c_batch):
            decoded_sentence = ''
            target_seq = np.zeros((1, 1, num_tokens), dtype='float32')
            target_seq[0, 0, token_index['&']] = 1.
            h, c = h_arr[np.newaxis, :], c_arr[np.newaxis, :]

            for _ in range(max_decoder_seq_length):
                out_tokens, h, c = decoder_model.predict([target_seq, h, c], verbose=1)
                sampled_i = int(np.argmax(out_tokens[0, -1, :]))
                sampled_char = reverse_target_char_index[sampled_i]
                if sampled_char == '$':
                    break
                decoded_sentence += sampled_char
                target_seq = np.zeros((1, 1, num_tokens), dtype='float32')
                target_seq[0, 0, sampled_i] = 1.

            results.append(decoded_sentence)

        print(f"  ✓ Finished batch up to sample {end}/{total}")

    # ─── 7) Save results ────────────────────────────────────────────
    os.makedirs('output', exist_ok=True)
    with open('output/results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("✔ All decoded results saved to output/results.pkl")

    return results
