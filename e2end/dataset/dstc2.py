import json, logging
import numpy as np
from collections import OrderedDict
import bisect
import pickle
from . import Vocabulary


logger = logging.getLogger(__name__)


class Dstc2DB:
    def __init__(self, filename, first_n=None):
        logger.info('\nLoading DB %s', filename)
        raw_data = json.load(open(filename))[:first_n]
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
        logger.info('\nLoaded DB %s.shape = %s', filename, self.table.shape)

    @property
    def column_names(self):
        return self._col_names

    @property
    def col_names_vocab(self):
        return self._col_name_vocab

    def get_col_idx(self, col_name):
        return self._col_name_vocab.get_i(col_name)

    def get_col_name(self, idx):
        return self._col_name_vocab.get_w(idx)

    @property
    def col_vocabs(self):
        return self._col_vocabs

    def matching_rows(self, col_val_dict):
        return self.table[self._row_mask(col_val_dict)]

    def _row_mask(self, col_val_dict):
        # See http://stackoverflow.com/questions/1962980/selecting-rows-from-a-numpy-ndarray
        return np.logical_and.reduce([self.table[:, c] == v for c, v in col_val_dict.items()])

    def matching_rest_names(self, col_val_dict):
        match_rows = self.matching_rows(col_val_dict)
        name_idx = self.get_col_idx('name')
        restaurant_names = match_rows[:, name_idx]
        return restaurant_names

    def get_col_vocab(self, col_name):
        idx = self.column_names.index(col_name)
        return self.col_vocabs[idx]

    @property
    def num_rows(self):
        return self.table.shape[0]

    @property
    def num_cols(self):
        return self.table.shape[1]

    @property
    def table(self):
        return self._table

    def extract_entities(self, sentence):
        '''Returns array of arrays of flags (0 or 1) indicating
        if a word is part if some value for in a column'''
        def mask_ent(sentence, vocab):
            mask = [0] * len(sentence)
            for i, w in enumerate(sentence):
                for ent in vocab.words:
                    e = ent.strip().split()
                    if w == e[0] and sentence[i:i + len(e)] == e:
                        logger.debug('found an entity %s in %s', ent, sentence)
                        mask[i:i + len(e)] = [1] * len(e)
                        break
            return mask

        return [mask_ent(sentence, vocab) for vocab in self.col_vocabs]


# FIXME implement caching, saving and loading
class Dstc2:
    '''
    Produces input, output labels for each turn.
    The input is the user AND system utterance from PREVIOUS turn.
    The output is ONLY system utterance from CURRENT turn.

    As a result for example the first turn has empty input and output the first system response.
    '''

    def __init__(self, filename, db, just_db=False,
            max_turn_len=None, max_dial_len=None, max_target_len=None, max_row_len=None,
            first_n=None, words_vocab=None, sample_unk=0):

        self.just_db = just_db
        self.restaurant_name_vocab_id = db.get_col_idx('name')
        logger.info('\nLoading dataset %s', filename)
        self.hello_token = hello_token = 'Hello'  # Default user history for first turn
        self.EOS = EOS = 'EOS'   # Symbol which the decoder should produce as last one'
        assert isinstance(db, Dstc2DB), type(db)
        raw_data = json.load(open(filename))
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
        logger.info('Dial length: %4d.\nDial turn len %4d.\n95-percentil %4d.\n', mdl, max_dial, perc95d)

        entities = [[db.extract_entities(turn) for turn in d] for d in dialogs]  

        logger.debug('Maximum decoder length increasing by 1, since targets are shifted by one')
        mdl += 1
        self._turn_lens_per_dialog = np.zeros((len(dialogs), mdl), dtype=np.int64)
        self._dials = np.zeros((len(dialogs), mdl, mtl), dtype=np.int64)
        self._word_ent = np.zeros((len(dialogs), mdl, len(db.column_names), mtl), dtype=np.int64)
        self._turn_lens = np.zeros((len(dialogs), mdl), dtype=np.int64)

        t = sorted([len(turn_target) for dialog_targets in targets for turn_target in dialog_targets])
        maxtarl, perc95t = t[-1], t[int(0.95 * len(s))]
        self._max_target_len = mtarl = (max_target_len or maxtarl)

        logger.info('Target len: %4d.\nMax target len %4d.\n95-percentil %4d.\n', maxtarl, mtarl, perc95t)
        self._turn_targets = ttarg = words_vocab.get_i(EOS) * np.ones((len(dialogs), mdl, mtarl), dtype=np.int64)
        self._turn_target_lens = np.zeros((len(dialogs), mdl), dtype=np.int64)
        self._word_speakers = w_spk = np.zeros((len(dialogs), mdl, mtl), dtype=np.int64)
        self._match_rows_props = np.zeros((len(dialogs), mdl, db.num_rows), dtype=np.int64)
        self._match_row_lens = np.zeros((len(dialogs), mdl), dtype=np.int64)

        tmp1, tmp2 = db.column_names + ['words'], db.col_vocabs + [words_vocab]
        self.target_vocabs = OrderedDict(zip(tmp1, tmp2))
        self.word_vocabs_uplimit = OrderedDict(
            zip(self.target_vocabs.keys(),
                np.cumsum([len(voc) for voc in self.target_vocabs.values()])))
        self.word_vocabs_downlimit = OrderedDict(
            zip(self.target_vocabs.keys(),
                [0] + list(self.word_vocabs_uplimit.values())[:-1]))

        dial_lens, this_max_row = [], 0
        for i, (d, spkss, entss, dtargss) in enumerate(zip(dialogs, speakers, entities, targets)):
            assert len(d) == len(dtargss)
            dial_len = 0
            logger.debug('Shifting targets and turns by one. First context is empty turn')
            d, spkss, entss = [[hello_token]] + d, [[usr]] + spkss, [db.extract_entities([hello_token])] + entss
            for j, (turn, spks, ents, targets) in enumerate(zip(d, spkss, entss, dtargss)):
                sys_word_ids, vocab_names = self._extract_vocab_ids(targets)

                restaurants = self._row_all_prop_match(sys_word_ids, vocab_names, db)
                num_match = restaurants.shape[0]
                this_max_row = max(num_match, this_max_row)

                if j > mdl or len(turn) > mtl or len(sys_word_ids) > mtarl or (max_row_len is not None and num_match > max_row_len):
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
                for k in range(num_match):
                    self._match_rows_props[i, j, k] = restaurants[k]
                self._match_row_lens[i, j] = num_match
                for k, w_id in enumerate(sys_word_ids):
                    ttarg[i, j, k] = w_id
            if dial_len > 0:
                dial_lens.append(dial_len)
            else:
                logger.debug('Discarding whole dialog: %d', i)

        self._max_match_rows = max_row_len or this_max_row
        self._match_rows_props = self._match_rows_props[:, :, :self._max_match_rows]

        logger.info('Max row len this set %d vs max_row_len %d', this_max_row, self._max_match_rows)
        self._dial_lens = np.array(dial_lens)

        for i, l in enumerate(self._dial_lens):
            self._dial_mask[i, :l] = np.ones((l,), dtype=np.int64)

        logger.info('\nLoaded dataset len(%s): %d', filename, len(self))

    # FIXME use it
    def _row_mention_match(self, word_ids, vocab_names, db):
        '''If a system response - word_ids contains an restaurant name, we know exact row which to output,
        but we can also output any row with the same properties which were mentioned.

        Args
            word_ids:
            vocab_names:
            db:

        Returns: numpy array representing mask of matching rows.'''
        if 'name' not in vocab_names:
            return np.array([])
        else:
            const = dict([(db.get_col_idx(vn), wid - self.word_vocabs_downlimit[vn]) for wid, vn in zip(word_ids, vocab_names) if vn in ['area', 'food', 'pricerange']])
            return db.matching_rest_names(const)

    def _row_all_prop_match(self, word_ids, vocab_names, db):
        '''If a system response - word_ids contains an restaurant name, we know exact row which to output,
        but we can also output any row with the same properties.

        Args
            word_ids:
            vocab_names:
            db:

        Returns: (num_match_rows, ids of restaurant names determining the matching rows).'''
        if 'name' not in vocab_names:
            return np.array([])
        else:
            name_idx = vocab_names.index('name')
            restaurant_idx = word_ids[name_idx] - self.word_vocabs_downlimit['name']
            name_vocab_id = self.restaurant_name_vocab_id  # id for name
            restaurant_row = db.matching_rows({name_vocab_id: restaurant_idx})
            assert len(restaurant_row) == 1, str(restaurant_row)
            restaurant_row = restaurant_row[0]
            col_idx = [db.get_col_idx(vn) for vn in ['area', 'food', 'pricerange']]
            filter_col_val = dict([(c, restaurant_row[c]) for c in col_idx])
            restaurants = db.matching_rest_names(filter_col_val)
            return restaurants

    def _extract_vocab_ids(self, target_words):
        '''Heuristic how to recognize named entities from DB in sentence and
        insert user their ids instead "regular words".'''
        skip_words_of_entity = 0
        target_ids, vocab_names = [], []
        for i, w in enumerate(target_words):
            if skip_words_of_entity > 0:
                skip_words_of_entity -= 1
                continue
            w_found = False
            for vocab_name, vocab in self.target_vocabs.items():
                if vocab_name == 'words':
                    continue
                for ent in vocab.words:
                    e = ent.strip().split()
                    if w == e[0] and target_words[i:i + len(e)] == e:
                        logger.debug('found an entity "%s" from column %s in target_words %s', ent, vocab_name, target_words)
                        skip_words_of_entity = len(e) - 1
                        w_id = self.get_target_surface_id(vocab_name, vocab, ent)
                        if self.just_db:
                            if vocab_name == 'name':
                                target_ids.append(w_id)
                                vocab_names.append(vocab_name)
                            else:
                                logger.debug('Reporting just restaurants names')
                        else:
                            target_ids.append(w_id)
                            vocab_names.append(vocab_name)
                        w_found = True
                        break
                if w_found:
                    break
            if not w_found:
                logger.debug('Target word "%s" treated as regular word', w)
                if self.just_db:
                    logger.debug('Skipping regular word in the targets')
                else:
                    target_ids.append(self.get_target_surface_id('words', self._vocab, w))
                    vocab_names.append('words')
        assert len(vocab_names) == len(target_ids)
        if self.just_db:
            target_ids.append(self.get_target_surface_id('words', self._vocab, self.EOS))
            vocab_names.append('words')
        return target_ids, vocab_names

    def get_target_surface_id(self, vocab_name, vocab, w):
        return self.word_vocabs_downlimit[vocab_name] + vocab.get_i(w)

    def get_target_surface(self, w_id):
        vocab_up_idx = list(self.word_vocabs_uplimit.values())
        vocab_id = bisect.bisect(vocab_up_idx, w_id)
        vocab_name = list(self.word_vocabs_uplimit.keys())[vocab_id]
        w_id_in_vocab = w_id - self.word_vocabs_downlimit[vocab_name]
        return vocab_name, self.target_vocabs[vocab_name].get_w(w_id_in_vocab)

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

    @property
    def gold_rows(self):
        return self._match_rows_props

    @property
    def gold_row_lens(self):
        return self._match_row_lens

    @property
    def max_row_len(self):
        return self._max_match_rows

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as r:
            return pickle.load(r)

    def save(self, filename):
        with open(filename, 'wb') as w:
            pickle.dump(self, w, protocol=2)

    def shuffle(self):
        raise NotImplementedError('I do not use it currently, I better raise the exception than update the list every time')
        # rng_state = np.random.get_state()
        # np.random.shuffle(self._dialogs)
        # np.random.set_state(rng_state)
