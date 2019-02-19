import re
import hazm
import os


max_file = 10 # maximum number of files which can retrieved frome each folder(label)



#vocab [label41] = label6
def make_topic_vocab (topic_file):
	vocab = {}
	for line in topic_file:
		line = line.strip ()
		if ':' in line: #this line is a 6topic
			label6 = line.replace (":", '')
		else:
			vocab [line] = label6
	return vocab



def write_to_string (input_text, label6, label41):
	#output_string6 = ""
	#output_string41 = ""
	output_string = ""
	wo_tag_text = re.sub('<[^<]+?>', '', input_text)
	wo_tag_text = re.sub('&nbsp;', '', wo_tag_text)
	sent_list = hazm.sent_tokenize(wo_tag_text)
	for sent in sent_list:
		word_list = hazm.word_tokenize (sent)
		word_tokenize_sent = ""
		for word in word_list:
			word_tokenize_sent += word + ' '
		output_string +=  word_tokenize_sent + "\t"
	output_string6 = output_string + "__label__" + label6 + '\n'
	output_string41 = output_string + "__label__" + label41 + '\n'
	return output_string6, output_string41


def write_to_file (input_file, output_file6, output_file41, label6, label41):
	input_text = input_file.read ()
	output_string6, output_string41 = write_to_string(input_text, label6, label41)
	# wo_tag_text = re.sub('<[^<]+?>', '',input_text)
	# wo_tag_text = re.sub ('&nbsp;', '', wo_tag_text)
	# sent_list = hazm.sent_tokenize (wo_tag_text)
	# for sent in sent_list:
	# 	word_list = hazm.word_tokenize (sent)
	# 	word_tokenize_sent = ""
	# 	for word in word_list:
	# 		word_tokenize_sent += word + ' '
	# 	output_file6.write (word_tokenize_sent + "\t")
	# 	output_file41.write (word_tokenize_sent + "\t")
	output_file6.write (output_string6)
	output_file41.write (output_string41)

if __name__ == '__main__':
	train41_path = "train{}-41.txt".format(max_file)
	test41_path = "test{}-41.txt".format(max_file)
	dev41_path = "dev{}-41.txt".format(max_file)
	train41_file = open(train41_path, 'wt')
	test41_file = open(test41_path, 'wt')
	dev41_file = open(dev41_path, 'wt')

	train6_path = "train{}-6.txt".format(max_file)
	test6_path = "test{}-6.txt".format(max_file)
	dev6_path = "dev{}-6.txt".format(max_file)
	train6_file = open(train6_path, 'wt')
	test6_file = open(test6_path, 'wt')
	dev6_file = open(dev6_path, 'wt')

	topic_file = open("topics")

	dir_name = "newsByCats"


	label_vocab = make_topic_vocab(topic_file)
	folder_name_list= os.listdir (dir_name)
	vocab_file = open ('label_6_41', 'wt')
	vocab_file.write(str(label_vocab))
	vocab_file.close ()

	for folder_name in folder_name_list:
		label41 = folder_name
		label6 = label_vocab [folder_name]
		file_name_list = os.listdir (dir_name + "/" + folder_name)
		total_file_cnt = min ((max_file, len (file_name_list)))
		train_file_cnt = int(0.8 * total_file_cnt) #80 percent of data goes to train, 10 % to develpment and the
													# other 10% to test
		dev_file_cnt = int(0.1 * total_file_cnt)
		test_file_cnt = total_file_cnt - (dev_file_cnt + train_file_cnt)
		for count, file_name in enumerate (file_name_list):
			#if count == 0:
			#	print (folder_name, file_name)
			input_file = open ("{}/{}/{}".format (dir_name, folder_name, file_name))
			if count < train_file_cnt :
				write_to_file (input_file, train6_file, train41_file, label6, label41)
			elif count < train_file_cnt + dev_file_cnt:
				write_to_file (input_file, dev6_file, dev41_file, label6, label41)
			elif count < total_file_cnt:
				write_to_file (input_file, test6_file, test41_file, label6, label41)
			input_file.close ()

	train41_file.close ()
	test41_file.close ()
	dev41_file.close ()

	train6_file.close ()
	test6_file.close ()
	dev6_file.close ()




