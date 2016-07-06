End-to-End Neural Dialog

Howto
-----

### Evaluate
- create validation dir with test set instead of dev set called e.g. `dstc2_test` and run
` ./demo.py  --load_dstc_dir dstc2_test --validate_to_dir test_valid --config path_to_training_config.json --load_model path_to_saved_model` 


Todo
----
- dynamic batch_size
- how to backprogate through `tf.assign` and multiple turns? tf.run_partial? - solved feeding whole history
    - problem too long
    - multiple turns are presented several times e.g. similar as nmt target words - longer dialogues gets bigger updates

Fix RL update
```
Xent updates for step    2999
Rein updates for step    3000
Training stopped after    3000 steps and    1.33 epochs. See logs for log/2016-06-02-16-08-21.827-encdec-row_targets-db_encoder-reinforce_fs_3000-evalf_match_ent_plusbleu-batch10/2016-06-02-16-08-21.827encdec-row_targets-db_encoder-reinforce_fs_3000-evalf_match_ent_plusbleu-batch10_traindir
Saving current state. Please wait!
Best model has reward   -6.67 form step    3000
Warning: Loading autoreload 2
import numpy as np      # Done
%matplotlib     # Done
Using matplotlib backend: agg
import matplotlib.pyplot as plt         # Done
Traceback (most recent call last):
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 644, in _do_call
    return fn(*args)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 628, in _run_fn
    session, None, feed_dict, fetch_list, target_list, None)
tensorflow.python.pywrap_tensorflow.StatusNotOK: Invalid argument: Incompatible shapes: [10,1562] vs. [10]
         [[Node: updates/reinforce_gradients/mul_1 = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/gpu:0"](updates/reinforce_gradients/mul, decoder/Squeeze_29)]]
         [[Node: updates/reinforce_grads/gradients/decoder/cond/att_decoder/attention_decoder/loop_function_24/embedding_lookup_grad/Reshape_1/_42046 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_264422_updates/reinforce_grads/gradients/decoder/cond/att_decoder/attention_decoder/loop_function_24/embedding_lookup_grad/Reshape_1", tensor_type=DT_INT64, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "./demo.py", line 31, in <module>
    training(c, sess, m, db, train, dev, c, train_writer, dev_writer)
  File "/ha/work/people/oplatek/e2end/e2end/training.py", line 170, in training
    tr_step_outputs = m.train_step(sess, input_fd, log_output=True)
  File "/ha/work/people/oplatek/e2end/e2end/model/__init__.py", line 368, in train_step
    sess.run([self.mixer_optimizer, self.update_expected_reward], feed_dict=train_dict)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 340, in run
    run_metadata_ptr)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 564, in _run
    feed_dict_string, options, run_metadata)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 637, in _do_run
    target_list, options, run_metadata)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py", line 659, in _do_call
    e.code)
tensorflow.python.framework.errors.InvalidArgumentError: Incompatible shapes: [10,1562] vs. [10]
         [[Node: updates/reinforce_gradients/mul_1 = Mul[T=DT_FLOAT, _device="/job:localhost/replica:0/task:0/gpu:0"](updates/reinforce_gradients/mul, decoder/Squeeze_29)]]
         [[Node: updates/reinforce_grads/gradients/decoder/cond/att_decoder/attention_decoder/loop_function_24/embedding_lookup_grad/Reshape_1/_42046 = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/cpu:0", send_device="/job:localhost/replica:0/task:0/gpu:0", send_device_incarnation=1, tensor_name="edge_264422_updates/reinforce_grads/gradients/decoder/cond/att_decoder/attention_decoder/loop_function_24/embedding_lookup_grad/Reshape_1", tensor_type=DT_INT64, _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
Caused by op 'updates/reinforce_gradients/mul_1', defined at:
  File "./demo.py", line 19, in <module>
    c, m, db, train, dev = parse_input()
  File "/ha/work/people/oplatek/e2end/e2end/utils.py", line 261, in parse_input
    m = Simple(c)
  File "/ha/work/people/oplatek/e2end/e2end/model/__init__.py", line 335, in __init__
    self._build_mixer_updates(dec_logitss, targets)
  File "/ha/work/people/oplatek/e2end/e2end/model/__init__.py", line 257, in _build_mixer_updates
    indicator, self.target_mask)]
  File "/ha/work/people/oplatek/e2end/e2end/model/__init__.py", line 256, in <listcomp>
    for r, l, i, w in zip(expected_rewards, dec_logitss,
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/ops/math_ops.py", line 518, in binary_op_wrapper
    return func(x, y, name=name)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/ops/gen_math_ops.py", line 1039, in mul
    return _op_def_lib.apply_op("Mul", x=x, y=y, name=name)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/ops/op_def_library.py", line 655, in apply_op
    op_def=op_def)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/framework/ops.py", line 2154, in create_op
    original_op=self._default_original_op, op_def=op_def)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/framework/ops.py", line 1154, in __init__
    self._traceback = _extract_stack()

try pudb
*** SyntaxError: invalid syntax
Warning: Loading autoreload 2
import numpy as np      # Done
%matplotlib     # Done
*** SyntaxError: invalid syntax
import matplotlib.pyplot as plt         # Done
> /ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/tensorflow/python/client/session.py(659)_do_call()
    658       raise errors._make_specific_exception(node_def, op, error_message,
--> 659                                             e.code)
    660       # pylint: enable=protected-access

ipdb>
^CError in atexit._run_exitfuncs:
Traceback (most recent call last):
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/IPython/core/interactiveshell.py", line 3229, in atexit_operations
    self.reset(new_session=False)
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/IPython/core/interactiveshell.py", line 1239, in reset
    self.displayhook.flush()
  File "/ha/home/oplatek/virtualenv/tensorflow-gpu/lib/python3.4/site-packages/IPython/core/displayhook.py", line 295, in flush
    gc.collect()
KeyboardInterrupt
```


- speed - speed up development by using GPU on OSX - https://gist.github.com/ageitgey/819a51afa4613649bd18
- use "autoencoding objective" as regularization/prove that the model is able to remember necessary pieces
- speed- robust learning - batch normalization - http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    - more useful http://www.r2rt.com/posts/implementations/2016-03-29-implementing-batch-normalization-tensorflow/
- use ipython notebook for TF graph caching
- use partial run for RL https://github.com/tensorflow/tensorflow/issues/672
- CF: zapnout ondruv spellchecker
- CF: zmenit cil dynamicky aby nerikali goaly vsechny najednou
- CF: proc konverzujou i za druhe
- CF: jak zarucit maximalne aby se userove neopakovali pro roundexpected_rewards
- pouzit softmax misto sigmoidy
- trenovat binarni klasifikatory pro vyber radku
- enforce decoder embeddings to be the same to DB and WORD embeddings (or make them close together - too complicated rather the same)
    - model:410 - TODO TEST
- reward function based on the dialog state
- filter only DB entries in targets [focus on DB]
- visualize decoding attentions
-  perplexity from TF translate. Rouge?
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
    - limit it for cluster - how to use one setting for cluster and for GPU?
    - limit parallel for cluster: `config = tf.ConfigProto(inter_op_paralelism_threads=4, intra_op_paralelism_threads=4)`
- data loading and model - do not recompute encoding - compute encoding for all prefixes in one dialog in one run
- debugging tools
    - visualize attentions - e.g. via https://www.youtube.com/watch?v=VcoVEvGEmFM
- rnn.rnn(sequence_length=sequence_length) use mask for the not used outputs
- investigate indexed_sliced
- Mixer can be easily (thanks for Jindra's implementation) used on top of xent
- implement batches first, than Batch normalization http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow?rq=1
- better tensorflow logging with `tf.merge_summary(tf.get_collection("summary_val"))`


Nice examples
-------------

- plain enc-dec  
inp 0008800,00: Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you? id like an expensive restaurant that serves bat food
dec 0008800,00: Ok I'm sorry but there is no restaurant serving australian food EOS
trg 0008800,00: I'm sorry but there is no restaurant serving basque food EOS

- plain enc-dec  
inp 0052000,00: anatolia serves turkish food in the moderate price range what is the phone number and address
dec 0052000,00: The phone number of meghna is 01223 727410 . EOS
trg 0052000,00: The phone number of anatolia is 01223 362372 and it is on 30 Bridge Street City Centre . EOS


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
    - ask about feed previous ODuska? Jindry? Filipa? - but first read http://arxiv.org/abs/1506.03099 [DONE] Fixed stupid bug
- initialize targets with EOS so it predicts always EOS after first one [DONE]
- bleu from nltk copy  [DONE]
- CF job-
    - clicknout on row DONE
