from __future__ import print_function

#from vocab_utils import Vocab
from .preprocess import write_to_file, write_to_string
from .HanDecoder import decode, load_model, load_data_stream, load_word_vocab

model_dir = 'app/HanClassifier/models'
suffix6 = 'run6-1'
suffix41 =  'run41-1'
word_vec_path = 'app/HanClassifier/data/glove/my_wiki.fa.vec'
in_path6 = "app/HanClassifier/tmpfile/tmp6"
in_path41 = "app/HanClassifier/tmpfile/tmp41"
input_path = "app/HanClassifier/tmpfile/tmpin"


def load_models (): #speed is not important for preprcoessing, but great speed :)
    #print ("***********load_models CALLED ************")
    word_vocab = load_word_vocab(word_vec_path)
    model_prefix6 = model_dir + "/Han.{}".format(suffix6)
    model_prefix41 = model_dir + "/Han.{}".format(suffix41)
    valid_graph6 ,sess6, label_vocab6, FLAGS6 = \
        load_model(model_prefix6,word_vocab)
    valid_graph41, sess41, label_vocab41, FLAGS41 = \
        load_model(model_prefix41, word_vocab)
    return  valid_graph6, sess6, label_vocab6, \
            valid_graph41, sess41, label_vocab41, word_vocab, FLAGS6


'''
speed of this function, must improve. because detection starts from here and it must be as fast as possible.
'''
def detect_class (input_text, load_params):
    valid_graph6, sess6, label_vocab6, \
    valid_graph41, sess41, label_vocab41, word_vocab, FLAGS = load_params
    # input_file = open (input_path, 'wt')
    # input_file.write(input_string)
    # input_file.close()
    # input_file = open(input_path)
    # input_file6 = open(in_path6, 'wt')
    # input_file41 = open (in_path41, 'wt')
    #write_to_file(input_file, input_file6, input_file41, label6 = 'l', label41 = 'l')
    input_string6, _ = write_to_string(input_text,label6='l', label41='l')
    #input_file6.close()
    #input_file41.close()
    model_prefix6 = model_dir + "/Han.{}".format(suffix6)
    model_prefix41 = model_dir + "/Han.{}".format(suffix41)
    #
    # print('Loading vocabs.')
    # word_vocab = Vocab(word_vec_path, fileformat='txt3')
    # label_vocab6 = Vocab(model_prefix6 + ".label_vocab", fileformat='txt2')
    # label_vocab41 = Vocab(model_prefix41 + ".label_vocab", fileformat='txt2')
    ''' one data stream is enough, we build it on string6 :)'''
    testDataStream = load_data_stream(input_string6, word_vocab, label_vocab6,FLAGS,
                               is_inpath_input_text=True)
    result6 = decode(model_prefix6, label_vocab6, valid_graph6, sess6,testDataStream)
    result41 = decode(model_prefix41, label_vocab41, valid_graph41, sess41, testDataStream)
    # result6 = decode(model_prefix6,in_path6,label_vocab6,
    #                  word_vocab, valid_graph6, sess6)
    # result41 = decode (model_prefix41, in_path41, label_vocab41,
    #                    word_vocab, valid_graph41, sess41)
    return result6, result41

if __name__ == '__main__':
    file = open ('app/HanClassifier/data/0.txt')
    input_string = file.read ()
    load_params = load_models()
    print (detect_class(input_string, load_params))

