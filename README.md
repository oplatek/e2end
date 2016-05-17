End-to-End Neural Dialog


Todo
----
- Should I normalize `sigmoid` attention by |DB| size?
- I should definitely normalize RL reward per turn (or later by dialog)
- Encoder
    - in addition to word embedding for each word concatenate `word or slot1, slot2, ..., slotK` embedding
    - in addition to word embedding for each word also concatenate `system or user` embedding or just features
- Encoder - Decoder glue
    - completely change the encoder in `tensorflow/python/ops/seq2seq.py:embedding_attention_seq2seq`
    - add supervised attention and mask for losses so I can focus on decoding `DB only`
- DB - after each turn select 1-K=2 rows to talk about
- decoder 
    - `tensorflow/python/ops/seq2seq.py:attention_decoder`
    - `_extract_argmax_and_embed` rewrite to `sample_and_embed`
        - `samle_and_embed` can remember visited states and mix the predicted distribution with `1 - (#visit_this_action / # visit_total)`
    - add `db_embedding` to attentions  `attn`: instead of `attns = attention(state)` use `attns_t_and_db = attention(state)`
        - `attns_t_and_db` is attention from encoder encoding words history context and last vector is `db_embedding`
            - We need to add `db_embedding` to `hidden_features`
            - `hidden_features = [h_u1_1, h_u1_2, ..., h_s1_1, h_s1_2, ..., ..., h_uk_9, h_uk_10, db_embedding1, db_embedding2]` 
            - `db_embeddings`: attention over rows and `tanh(row_is_selected_activation)`
    - decode one of `1 ... total = |words_vocab| + |slot1_vocab| + ... + |slotK_vocab|` and also a `word or slot1, slot2, slotK` embedding
        - selecting the index to vocabularies from `k from 1 ... total`: 
        - `cumsummax = [|voc1|, |voc1| + |voc2|, ... , |voc1|+...+|vocK|]`
        - `cumsummin = [0] + cumsummax[:-1]`
        - `mask_vocabs = min(1, max(i - cumsummin, 0) * max(cumsummax - i, 0))`
        - `vocab_index = argmax(mask_vocabs)`

- RL
    - sampling for RL: move it to TF: See [issue/PR 565](https://github.com/tensorflow/tensorflow/pull/2093/files)
    - as a baseline use parameters few epochs back

- parallelization `NUM_THREADS = 20;  sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))`
- data loading and model - do not recompute encoding - compute encoding for all prefixes in one dialog in one run
- debugging tools
    - visualize attentions - e.g. via https://www.youtube.com/watch?v=VcoVEvGEmFM
- rnn.rnn(sequence_length=sequence_length) use mask for the not used outputs
- investigate indexed_sliced
- Mixer can be easily (thanks for Jindra's inspiration) used on top of xent
- better options loader combine with configs from Alex and https://pypi.python.org/pypi/json-cfg
