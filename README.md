End-to-End Neural Dialog


Todo
----
- ask about feed previous ODuska? Jindry? Filipa? - but first read http://arxiv.org/abs/1506.03099
- visualize decoding attentions
- copy bleu from Jindras and copy perplexity from TF translate. Rouge?
- filter only DB entries in targets [focus on DB]
- enforce decoder embeddings to be the same to DB and WORD embeddings (or make them close together - too complicated rather the same)
    - model:410
- add new loss in addition to 'standard' loss (which can be used to compute perplexity - actually computing perplexity would be nicer)
    - valid row reward 
        - check that the presented information form a row
        - study the responses so they do not talk about multiple rows - need to study in advance
    - matching properties
        - 0.5 if there is correctly no DB property
        - 0 if there there should be a DB property but it is not
        - 0 if there is a DB property but it shouldn't be
        - if db properties number match return: 0.5 + (0.5 * correct_properties / all_properties)
            - the 0.5 addend easy to implement in tf - check vocab indexes
            - the rest hard - must determine what if we care about the property and what are the matching
                - E.g. we do not care for names if we were not ask for it but we were care for food_type=chinese because we were asked for it 
- Should I normalize `sigmoid` attention by |DB| size?
- I should definitely normalize RL reward per turn (or later by dialog)
- RL
    - sampling for RL: move it to TF: See [issue/PR 565](https://github.com/tensorflow/tensorflow/pull/2093/files)
    - as a baseline use parameters few epochs back
- Encoder - Decoder glue
    - completely change the encoder in `tensorflow/python/ops/seq2seq.py:embedding_attention_seq2seq`
    - add supervised attention and mask for losses so I can focus on decoding `DB only`
- DB - after each turn select 1-K=2 rows to talk about
- decoder 
    - `tensorflow/python/ops/seq2seq.py:attention_decoder`
    - `_extract_argmax_and_embed` rewrite to `sample_and_embed`
        - `samle_and_embed` can remember visited states and mix the predicted distribution with `1 - (#visit_this_action / # visit_total)`


- parallelization `NUM_THREADS = 20;  sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=NUM_THREADS))`
- data loading and model - do not recompute encoding - compute encoding for all prefixes in one dialog in one run
- debugging tools
    - visualize attentions - e.g. via https://www.youtube.com/watch?v=VcoVEvGEmFM
- rnn.rnn(sequence_length=sequence_length) use mask for the not used outputs
- investigate indexed_sliced
- Mixer can be easily (thanks for Jindra's inspiration) used on top of xent
- better options loader combine with configs from Alex and https://pypi.python.org/pypi/json-cfg


Nice examples
-------------

enc-dec problem 
inp 0008800,00: Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you? id like an expensive restaurant that serves bat food
dec 0008800,00: Ok I'm sorry but there is no restaurant serving australian food EOS
trg 0008800,00: I'm sorry but there is no restaurant serving basque food EOS


DONE
---
- Encoder
    - in addition to word embedding for each word concatenate `word or slot1, slot2, ..., slotK` embedding - DONE
    - in addition to word embedding for each word also concatenate `system or user` embedding or just features - DONE
- Decoder
    - add `db_embedding` to attentions  `attn`: instead of `attns = attention(state)` use `attns_t_and_db = attention(state)` [DONE]
        - `attns_t_and_db` is attention from encoder encoding words history context and last vector is `db_embedding` [DONE]
            - We need to add `db_embedding` to `hidden_features` [DONE]
            - `hidden_features = [h_u1_1, h_u1_2, ..., h_s1_1, h_s1_2, ..., ..., h_uk_9, h_uk_10, db_embedding1]` [DONE]
            - `db_embeddings`: attention over rows and `tanh(row_is_selected_activation) to encoded state` [DONE]
    - decode one of `1 ... total = |words_vocab| + |slot1_vocab| + ... + |slotK_vocab|` and also a `word or slot1, slot2, slotK` embedding [DONE]
        - selecting the index to vocabularies from `k from 1 ... total`: 
        - `cumsummax = [|voc1|, |voc1| + |voc2|, ... , |voc1|+...+|vocK|]`
        - `cumsummin = [0] + cumsummax[:-1]`
        - `mask_vocabs = min(1, max(i - cumsummin, 0) * max(cumsummax - i, 0))`
        - `vocab_index = argmax(mask_vocabs)`
