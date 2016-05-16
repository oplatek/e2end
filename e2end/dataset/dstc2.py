import json, logging
import numpy as np
from collections import OrderedDict
import bisect

from . import Vocabulary


logger = logging.getLogger(__name__)


class Dstc2DB:
    def __init__(self, filename, first_n=None):
        self._raw_data = raw_data = json.load(open(filename))[:first_n]
        self._col_names = col_names = sorted(list(set([k for r in raw_data for k in r.keys()])))
        self._col_name_vocab = Vocabulary([], extra_words=col_names, unk=None)
        self._col_vocabs = col_vocabs = []
        for c in col_names:
            col_occur = [r[c] for r in raw_data]
            col_vocabs.append(Vocabulary(col_occur, max_items=len(col_occur)))
        self._table = table = np.empty((len(raw_data), len(col_names)), dtype=np.int64)
        for j, (cv, cn) in enumerate(zip(col_vocabs, col_names)):
            for i, r in enumerate(raw_data):
                table[i, j] = cv.get_i(r[cn])

    @property
    def column_names(self):
        return self._col_names

    @property
    def col_names_vocab(self):
        self._col_name_vocab

    def get_col_idx(self, col_name):
        return self.column_names_vocab.get_i(col_name)

    def get_col_name(self, idx):
        return self.column_names_vocab.get_w(idx)

    @property
    def col_vocabs(self):
        return self._col_vocabs

    def get_col_vocab(self, col_name):
        idx = self.column_names.index(col_name)
        return self.col_vocabs[idx]

    @property
    def table(self):
        return self._table

    def extract_entities(self, sentence):
        '''Returns array of arrays of flags (0 or 1) indicating
        if a word is part if some value for in a column'''
        def mask_ent(sentence, vocab):
            mask = [0] * len(sentence)
            for i, w in enumerate(sentence):
                for ent in vocab.words():
                    e = ent.strip().split()
                    if w == e[0] and sentence[i:i + len(e)] == e:
                        logger.debug('found an entity %s in %s', ent, sentence)
                        mask[i:i + len(e)] = [1] * len(e)
                        break
            return mask

        return [mask_ent(sentence, vocab) for vocab in self.col_vocabs]


class Dstc2:
    '''
    Produces input, output labels for each turn.
    The input is the user AND system utterance from PREVIOUS turn.
    The output is ONLY system utterance from CURRENT turn.

    As a result for example the first turn has empty input and output the first system response.
    '''

    def __init__(self, filename, db,
            max_turn_len=None, max_dial_len=None, max_target_len=None,
            first_n=None, words_vocab=None, labels_vocab=None, sample_unk=0):

        self.hello_token = hello_token = 'Hello'  # Default user history for first turn
        self.EOS = EOS = 'EOS'   # Symbol which the decoder should produce as last one'
        assert isinstance(db, Dstc2DB), type(db)
        self._raw_data = raw_data = json.load(open(filename))
        self._first_n = min(first_n, len(raw_data)) if first_n else len(raw_data)
        dialogs = [[(turn[0] + ' ' + turn[1]).strip().split() for turn in dialog] for dialog in raw_data]
        self._speak_vocab = Vocabulary([], extra_words=['usr', 'sys'], unk=None)
        usr, ss = self._speak_vocab.get_i('usr'), self._speak_vocab.get_i('sys')
        speakers = [[[ss] * len(turn[0].strip().split()) + [usr] * len(turn[1].strip().split()) for turn in dialog] for dialog in raw_data]

        labels = [[turn[4] for turn in dialog] for dialog in raw_data]
        assert len(dialogs) == len(labels), '%s vs %s' % (dialogs, labels)
        targets = [[(turn[0]).strip().split() + [EOS] for turn in dialog] for dialog in raw_data]

        dialogs, labels, speakers, targets = dialogs[:first_n], labels[:first_n], speakers[:first_n], targets[:first_n]

        self._vocab = words_vocab = words_vocab or Vocabulary([w for turns in dialogs for turn in turns for w in turn], extra_words=[hello_token, EOS], unk='UNK')

        s = sorted([len(t) for turns in dialogs for t in turns])
        max_turn, perc95t = s[-1], s[int(0.95 * len(s))]
        self._max_turn_len = mtl = max_turn_len or max_turn
        logger.info('Turn length: %4d.\nMax turn len %4d.\n95-percentil %4d.\n', mtl, max_turn, perc95t)
        d = sorted([len(d) for d in dialogs])
        max_dial, perc95d = d[-1], d[int(0.95 * len(d))]
        self._max_dial_len = mdl = max_dial_len or max_dial 
        logger.info('Dial length: %4d.\Dial turn len %4d.\n95-percentil %4d.\n', mdl, max_dial, perc95d)

        entities = [[db.extract_entities(turn) for turn in d] for d in dialogs]  

        logger.debug('Maximum decoder length increasing by 1, since targets are shifted by one')
        mdl += 1
        self._turn_lens_per_dialog = np.zeros((len(dialogs), mdl), dtype=np.int64)
        self._dials = np.zeros((len(dialogs), mdl, mtl), dtype=np.int64)
        self._word_ent = np.zeros((len(dialogs), mdl, len(db.column_names), mtl), dtype=np.int64)
        self._turn_lens = np.zeros((len(dialogs), mdl), dtype=np.int64)

        t = sorted([len(turn_target) for dialog_targets in targets for turn_target in dialog_targets])
        maxtarl, perc95t = t[-1], t[int(0.95 * len(s))]
        self._max_target_len = mtarl = max_target_len or maxtarl
        logger.info('Target len: %4d.\nMax target len %4d.\n95-percentil %4d.\n', maxtarl, mtarl, perc95t)
        self._turn_targets = ttarg = np.zeros((len(dialogs), mdl, mtarl), dtype=np.int64)
        self._turn_target_lens = np.zeros((len(dialogs), mdl), dtype=np.int64)
        self._word_speakers = w_spk = np.zeros((len(dialogs), mdl, mtl), dtype=np.int64)

        tmp1, tmp2 = db.column_names + ['words'], db.col_vocabs + [words_vocab]
        self._target_vocabs = OrderedDict(zip(tmp1, tmp2))
        self.word_vocabs_uplimit = OrderedDict(
            zip(self._target_vocabs.keys(),
                np.cumsum([len(voc) for voc in self._target_vocabs.values()])))
        self.word_vocabs_downlimit = OrderedDict(
            zip(self._target_vocabs.keys(),
                [0] + list(self.word_vocabs_uplimit.values())[:-1]))

        dial_lens = []
        for i, (d, spkss, entss, dtargss) in enumerate(zip(dialogs, speakers, entities, targets)):
            assert len(d) == len(dtargss)
            dial_len = 0
            logger.debug('Shifting targets and turns by one. First context is empty turn')
            d, spkss, entss = [[hello_token]] + d, [[usr]] + spkss, [db.extract_entities([hello_token])] + entss
            for j, (turn, spks, ents, targets) in enumerate(zip(d, spkss, entss, dtargss)):
                sys_word_ids = self._extract_vocab_ids(targets)
                if j > mdl or len(turn) > mtl or len(sys_word_ids) > mtarl:
                    logger.debug("Keep prefix of turns, discard following turns because:"
                        "a) num_turns too big "
                        "b) current turn too long. "
                        "c) current target too long.")
                    break
                else:
                    dial_len += 1
                assert len(turn) == len(spks), str((len(turn), len(spks), turn, spks))
                self._turn_lens[i, j] = len(turn)
                for k, (w, s_id) in enumerate(zip(turn, spks)):
                    self._dials[i, j, k] = words_vocab.get_i(w, unk_chance_smaller=sample_unk)
                    w_spk[i, j, k] = s_id
                    for l, e in enumerate(ents):
                        self._word_ent[i, j, l, k] = e[k]

                self._turn_target_lens[i, j] = len(sys_word_ids)
                for k, w_id in enumerate(sys_word_ids):
                    ttarg[i, j, k] = w_id
            if dial_len > 0:
                dial_lens.append(dial_len)
            else:
                logger.debug('Discarding whole dialog: %d', i)

        self._dial_lens = np.array(dial_lens)

    def _extract_vocab_ids(self, target_words):
        '''Heuristic how to recognize named entities from DB in sentence and
        insert user their ids instead "regular words".'''
        skip_words_of_entity = 0
        target_ids = []
        for i, w in enumerate(target_words):
            if skip_words_of_entity > 0:
                skip_words_of_entity -= 1
                continue
            w_found = False
            for vocab_name, vocab in self._target_vocabs.items():
                if vocab_name == 'words':
                    continue
                for ent in vocab.words():
                    e = ent.strip().split()
                    if w == e[0] and target_words[i:i + len(e)] == e:
                        logger.debug('found an entity "%s" from column %s in target_words %s', ent, vocab_name, target_words)
                        skip_words_of_entity = len(e) - 1
                        w_id = self.get_target_surface_id(vocab_name, vocab, ent)
                        target_ids.append(w_id)
                        w_found = True
                        break
                if w_found:
                    break
            if not w_found:
                logger.debug('Target word "%s" treated as regular word', w)
                target_ids.append(self.get_target_surface_id('words', self._vocab, w))
        return target_ids

    def get_target_surface_id(self, vocab_name, vocab, w):
        return self.word_vocabs_downlimit[vocab_name] + vocab.get_i(w)

    def get_target_surface(self, w_id):
        vocab_up_idx = list(self.word_vocabs_uplimit.values())
        vocab_id = bisect.bisect(vocab_up_idx, w_id)
        vocab_name = list(self.word_vocabs_uplimit.keys())[vocab_id]
        w_id_in_vocab = w_id - self.word_vocabs_downlimit[vocab_name]
        return vocab_name, self._target_vocabs[vocab_name].get_w(w_id_in_vocab)

    def __len__(self):
        '''Number of dialogs valid in other variables'''
        return len(self._dial_lens)

    @property
    def words_vocab(self):
        return self._vocab

    @property
    def speakers_vocab(self):
        return self._speak_vocab

    @property
    def max_dial_len(self):
        return self._max_dial_len

    @property
    def max_turn_len(self):
        return self._max_turn_len

    @property
    def max_target_len(self):
        return self._max_target_len

    @property
    def dialogs(self):
        '''Returns np.array with shape [#dialogs, max_dial_len, max_turn_len]
        which stores words in dialogues history indexed by dialogs and turns.'''
        return self._dials

    @property
    def word_speakers(self):
        '''Returns np.array of shape [#dialogs, max_dial_len, max_turn_len]
        which stores for each word an id of the word's speaker.''' 
        return self._word_speakers

    @property
    def word_entities(self):
        '''Returns np.array of shape [#dialogs, max_dial_len, max_turn_len, #db_columns]
        which stores 1,0 flags whether a word in dialog history is part of named entity
        from a DB column.'''
        return self._word_ent

    @property
    def turn_lens(self):
        return self._turn_lens

    @property
    def dial_lens(self):
        return self._dial_lens

    @property
    def att_mask(self):
        return self._att_mask

    @property
    def turn_targets(self):
        return self._turn_targets

    @property
    def turn_target_lens(self):
        return self._turn_target_lens

    def shuffle(self):
        rng_state = np.random.get_state()
        np.random.shuffle(self._dialogs)
        np.random.set_state(rng_state)
        np.random.shuffle(self._word_speakers)
        np.random.set_state(rng_state)
        np.random.shuffle(self._word_ent)
        np.random.set_state(rng_state)
        np.random.shuffle(self._turn_lens)
        np.random.set_state(rng_state)
        np.random.shuffle(self._att_mask)
        np.random.set_state(rng_state)
        np.random.shuffle(self._turn_targets)
        np.random.set_state(rng_state)
        np.random.shuffle(self._turn_target_lens)
        np.random.set_state(rng_state)
        np.random.shuffle(self.dial_lens)
