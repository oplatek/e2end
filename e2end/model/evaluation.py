from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
import tensorflow as tf
import numpy as np


bleu_smoothing = SmoothingFunction(epsilon=0.01).method1


def get_bleus(referencess, wordss):
    '''Return bleu using nltk and 0.0 for empty decoded sequnces'''
    return [sentence_bleu([r], s, smoothing_function=bleu_smoothing) if s else 0.0 for r, s in zip(referencess, wordss)]


def tf_lengths2mask2d(lengths, max_len):
    batch_size = lengths.get_shape().as_list()[0]
    # Filled the row of 2d array with lengths
    lengths_transposed = tf.expand_dims(lengths, 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, max_len])

    # Filled the rows of 2d array 0, 1, 2,..., max_len ranges 
    rng = tf.to_int64(tf.range(0, max_len, 1))
    range_row = tf.expand_dims(rng, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    # Use the logical operations to create a mask
    return tf.to_float(tf.less(range_tiled, lengths_tiled))


def tf_trg_word2vocab_id(wt_arr, vocabs_cum_start_idx_low, vocabs_cum_start_idx_up, name='tf_trg_word2vocab_id'):
    '''Map words + [EOS] to word vocab id, other entities to their
    column-slot vocab id'''
    with tf.variable_scope(name):
        res = []
        for wt in wt_arr:
            wte = tf.expand_dims(wt, 1)
            start_idx_ok = tf.greater_equal(wte, vocabs_cum_start_idx_low)
            end_idx_ok = tf.less(wte, vocabs_cum_start_idx_up)
            idx_ok_mask = tf.logical_and(start_idx_ok, end_idx_ok)
            tf.Print(idx_ok_mask, [idx_ok_mask], message='debug idx_ok_mask', name='debug')
            batch_vocab_id = tf.squeeze(tf.slice(tf.where(idx_ok_mask), [0, 1], [-1, 1]), [1])
            res.append(batch_vocab_id)
        return res


def row_acc_cov(np_pred, np_gold):
    eqv = len(np.intersect1d(np_pred, np_gold))
    pred_num = len(np_pred)
    gold_num = len(np_gold)
    acc = eqv / pred_num if pred_num > 0 else 1.0
    cov = eqv / gold_num if gold_num > 0 else 1.0
    return acc, cov


def rouge2(decoded, references):
    pass


def split_facts_syntax(vocab_ids, words_vocab_id):
    db_entities = [vocab_id for vocab_id in vocab_ids if vocab_id != words_vocab_id]
    words = [vocab_id for vocab_id in vocab_ids if vocab_id != words_vocab_id]
    return db_entities, words


def bleu_4(decoded, references):
    listed_references = [[s] for s in references]

    bleu_4 = \
        100 * corpus_bleu(listed_references, decoded,
                      weights=[0.25, 0.25, 0.25, 0.25],
                      smoothing_function=bleu_smoothing)
    return bleu_4


def bleu_1(decoded, references):
    listed_references = [[s] for s in references]

    bleu_1 = \
        100 * corpus_bleu(listed_references, decoded,
                      weights=[1.0, 0, 0, 0],
                      smoothing_function=bleu_smoothing)
    return bleu_1


def bleu_4_dedup(decoded, references):
    listed_references = [[s] for s in references]
    deduplicated_sentences = []

    for sentence in decoded:
        last_w = None
        dedup_snt = []

        for word in sentence:
            if word != last_w:
                dedup_snt.append(word)
                last_w = word

        deduplicated_sentences.append(dedup_snt)

    bleu_4 = \
        100 * corpus_bleu(listed_references, deduplicated_sentences,
                      weights=[0.25, 0.25, 0.25, 0.25],
                      smoothing_function=bleu_smoothing)
    return bleu_4
