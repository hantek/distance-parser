import os
import pickle

import torch

from helpers import *


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print('Number of words:', len(self.idx2word))
        return len(self.idx2word)

    def class_weight(self):
        frq = [self.word2frq[self.idx2word[i]] for i in range(len(self.idx2word))]
        frq = numpy.array(frq).astype('float')
        weight = numpy.sqrt(frq.max() / frq)
        weight = numpy.clip(weight, a_min=0., a_max=5.)

        return weight


class PTBLoader(object):
    '''Data path is assumed to be a directory with
       pkl files and a corpora subdirectory.
    '''
    def __init__(self, data_path=None, use_glove=False):
        assert data_path is not None
        # make path available for nltk
        nltk.data.path.append(data_path)
        dict_filepath = os.path.join(data_path, 'dict.pkl')
        data_filepath = os.path.join(data_path, 'parsed.pkl')

        print("loading dictionary ...")
        self.dictionary = pickle.load(open(dict_filepath, "rb"))
        if use_glove:
            glove_filepath = os.path.join(data_path, 'ptb_glove.npy')
            print("loading preprocessed glove file ...")
            f_we = open(glove_filepath, 'rb')
            self.wordembed_matrix = numpy.load(f_we)
            f_we.close()
        else:
            self.wordembed_matrix = None

        # build tree and distance
        print("loading tree and distance ...")
        file_data = open(data_filepath, 'rb')
        self.train, self.arc_dictionary, self.stag_dictionary = pickle.load(file_data)
        self.valid = pickle.load(file_data)
        self.test = pickle.load(file_data)
        file_data.close()

    def batchify(self, dataname, batch_size, sort=False):
        sents, trees = None, None
        if dataname == 'train':
            idxs, tags, stags, arcs, distances, sents, trees = self.train
        elif dataname == 'valid':
            idxs, tags, stags, arcs, distances, _, _ = self.valid
        elif dataname == 'test':
            idxs, tags, stags, arcs, distances, _, _ = self.test
        else:
            raise 'need a correct dataname'

        assert len(idxs) == len(distances)
        assert len(idxs) == len(tags)

        bachified_idxs, bachified_tags, bachified_stags, \
        bachified_arcs, bachified_dsts, \
            = [], [], [], [], []
        bachified_sents, bachified_trees = [], []
        for i in range(0, len(idxs), batch_size):
            if i + batch_size >= len(idxs): continue

            if sents is not None:
                bachified_sents.append(sents[i: i + batch_size])
                bachified_trees.append(trees[i: i + batch_size])

            extracted_idxs = idxs[i: i + batch_size]
            extracted_tags = tags[i: i + batch_size]
            extracted_stags = stags[i: i + batch_size]

            extracted_arcs = arcs[i: i + batch_size]
            extracted_dsts = distances[i: i + batch_size]

            longest_idx = max([len(i) for i in extracted_idxs])
            longest_arc = longest_idx - 1

            minibatch_idxs, minibatch_tags, minibatch_stags, \
            minibatch_arcs, minibatch_dsts, \
                = [], [], [], [], []
            for idx, tag, stag, \
                arc, dst \
                    in zip(extracted_idxs, extracted_tags, extracted_stags,
                           extracted_arcs, extracted_dsts):
                padded_idx = idx + [-1] * (longest_idx - len(idx))
                padded_tag = tag + [0] * (longest_idx - len(tag))
                padded_stag = stag + [0] * (longest_idx - len(stag))

                padded_arc = arc + [0] * (longest_arc - len(arc))
                padded_dst = dst + [0] * (longest_arc - len(dst))

                minibatch_idxs.append(padded_idx)
                minibatch_tags.append(padded_tag)
                minibatch_stags.append(padded_stag)

                minibatch_arcs.append(padded_arc)
                minibatch_dsts.append(padded_dst)

            minibatch_idxs = torch.LongTensor(minibatch_idxs)
            minibatch_tags = torch.LongTensor(minibatch_tags)
            minibatch_stags = torch.LongTensor(minibatch_stags)

            minibatch_arcs = torch.LongTensor(minibatch_arcs)
            minibatch_dsts = torch.FloatTensor(minibatch_dsts)

            bachified_idxs.append(minibatch_idxs)
            bachified_tags.append(minibatch_tags)
            bachified_stags.append(minibatch_stags)

            bachified_arcs.append(minibatch_arcs)
            bachified_dsts.append(minibatch_dsts)

        if sents is not None:
            return bachified_idxs, bachified_tags, bachified_stags, \
                bachified_arcs, bachified_dsts, \
                bachified_sents, bachified_trees
        return bachified_idxs, bachified_tags, bachified_stags, \
               bachified_arcs, bachified_dsts

