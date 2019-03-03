import numpy as np
import re
import sys





def pad_2d_matrix(in_val, max_length=None, dtype=np.int32):
    if max_length is None: max_length = np.max([len(cur_in_val) for cur_in_val in in_val])
    batch_size = len(in_val)
    out_val = np.zeros((batch_size, max_length), dtype=dtype)
    for i in range(batch_size):
        cur_in_val = in_val[i]
        kept_length = len(cur_in_val)
        if kept_length>max_length: kept_length = max_length
        out_val[i,:kept_length] = cur_in_val[:kept_length]
    return out_val


def make_indices(text, word_vocab, max_sent_length):
    item = re.split("\t", text) #Sentences Are Splitted with tab
    word_idx_list = []
    for i in range (len (item)):
        word_idx =word_vocab.to_index_sequence(item[i])
        if len(word_idx) > max_sent_length:
            word_idx = word_idx[:max_sent_length]
        word_idx_list.append (word_idx)
    return word_idx_list

def make_idx (label_vocab, label, word_vocab, text, max_sent_length):
    label_id = label_vocab.getIndex(label)
    word_idx = make_indices(text, word_vocab,max_sent_length)
    return (label_id, word_idx)


class HanDataStream(object):
    def __init__(self, inpath, word_vocab, label_vocab,
                 isShuffle, isLoop, max_sent_length, is_inpath_input_text = False):
        self.instances = []
        if is_inpath_input_text == False: #inpath is not input text and is file path which containes input text.
            infile = open(inpath, 'rt')
        else: #inpath is input text :) this function used in HanCaller where the input is text string, not file
            infile = inpath.split ('\n')
            #print (infile)
        for line in infile:
            if  (len (line) < 5): continue
            if sys.version_info[0] < 3:
                line = line.decode('utf-8').strip()
            else:
                line = line.strip()
            items = re.split("__label__", line) #to split text and label
            label = items[1]
            text = items[0]
            label_id, word_idx = \
                make_idx(label_vocab, label, word_vocab, text, max_sent_length)
            #word_idx is list of word indexs of each sentence in the text [SentCnt, SentLen]
            sents_length = [len(cur_word_idx) for cur_word_idx in word_idx]
            self.instances.append((label, text, label_id,
                                   pad_2d_matrix(word_idx), np.array(sents_length)))
        if is_inpath_input_text == False:
            infile.close()
        self.num_instances = len(self.instances)

        self.index_array = np.arange(self.num_instances)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0
        #print('DataStream Finished')


    def get_cur_pointer (self):
        return self.cur_pointer

    def get_num_batches (self, batch_size):
        if self.num_instances % batch_size == 0:
            return self.num_instances // batch_size
        else:
            return 1 + self.num_instances // batch_size

    def nextInstance(self):
        if self.cur_pointer >= self.num_instances:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_instance = self.instances[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_instance, self.index_array[self.cur_pointer - 1]

    def reset(self):
        # if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_instance(self):
        return self.num_instances

    def get_instance(self, i):
        if i >= self.num_instances: return None
        return self.instances[i]
