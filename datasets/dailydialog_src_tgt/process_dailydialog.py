import json
import sys
import re
from nltk.tokenize import TweetTokenizer


def clean_str(txt):
    # author: Xiang Gao @ Microsoft Research, Oct 2018
    # clean and tokenize natural language text

	#print("in=[%s]" % txt)
	txt = txt.lower()
	txt = re.sub('^',' ', txt)
	txt = re.sub('$',' ', txt)

	# url and tag
	words = []
	for word in txt.split():
		i = word.find('http') 
		if i >= 0:
			word = word[:i] + ' ' + '__url__'
		words.append(word.strip())
	txt = ' '.join(words)

	# remove markdown URL
	txt = re.sub(r'\[([^\]]*)\] \( *__url__ *\)', r'\1', txt)

	# remove illegal char
	txt = re.sub('__url__','URL',txt)
	txt = re.sub(r"[^A-Za-z0-9():,.!?\"\']", " ", txt)
	txt = re.sub('URL','__url__',txt)	

	# contraction
	add_space = ["'s", "'m", "'re", "n't", "'ll","'ve","'d","'em"]
	tokenizer = TweetTokenizer(preserve_case=False)
	txt = ' ' + ' '.join(tokenizer.tokenize(txt)) + ' '
	txt = txt.replace(" won't ", " will n't ")
	txt = txt.replace(" can't ", " can n't ")
	for a in add_space:
		txt = txt.replace(a+' ', ' '+a+' ')

	txt = re.sub(r'^\s+', '', txt)
	txt = re.sub(r'\s+$', '', txt)
	txt = re.sub(r'\s+', ' ', txt) # remove extra spaces
	
	#print("out=[%s]" % txt)
	return txt


def clean_en_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    #  text = re.sub(r"ph\.d", "phd", text)
    #  text = re.sub(r" e g ", " eg ", text)
    #  text = re.sub(r" fb ", " facebook ", text)
    #  text = re.sub(r"facebooks", " facebook ", text)
    #  text = re.sub(r"facebooking", " facebook ", text)
    #  text = re.sub(r" usa ", " america ", text)
    #  text = re.sub(r" us ", " america ", text)
    #  text = re.sub(r" u s ", " america ", text)
    #  text = re.sub(r" U\.S\. ", " america ", text)
    #  text = re.sub(r" US ", " america ", text)
    #  text = re.sub(r" American ", " america ", text)
    #  text = re.sub(r" America ", " america ", text)
    #  text = re.sub(r" mbp ", " macbook-pro ", text)
    #  text = re.sub(r" mac ", " macbook ", text)
    #  text = re.sub(r"macbook pro", "macbook-pro", text)
    #  text = re.sub(r"macbook-pros", "macbook-pro", text)
    #  text = re.sub(r" 1 ", " one ", text)
    #  text = re.sub(r" 2 ", " two ", text)
    #  text = re.sub(r" 3 ", " three ", text)
    #  text = re.sub(r" 4 ", " four ", text)
    #  text = re.sub(r" 5 ", " five ", text)
    #  text = re.sub(r" 6 ", " six ", text)
    #  text = re.sub(r" 7 ", " seven ", text)
    #  text = re.sub(r" 8 ", " eight ", text)
    #  text = re.sub(r" 9 ", " nine ", text)
    #  text = re.sub(r"googling", " google ", text)
    #  text = re.sub(r"googled", " google ", text)
    #  text = re.sub(r"googleable", " google ", text)
    #  text = re.sub(r"googles", " google ", text)
    #  text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    #  text = re.sub(r"\+", " + ", text)
    #  text = re.sub(r"'", " ", text)
    #  text = re.sub(r"-", " - ", text)
    #  text = re.sub(r"/", " / ", text)
    #  text = re.sub(r"\\", " \ ", text)
    #  text = re.sub(r"=", " = ", text)
    #  text = re.sub(r"\^", " ^ ", text)
    #  text = re.sub(r":", " : ", text)
    #  text = re.sub(r"\.", " . ", text)
    #  text = re.sub(r",", " , ", text)
    #  text = re.sub(r"\?", " ? ", text)
    #  text = re.sub(r"!", " ! ", text)
    #  text = re.sub(r"\"", " \" ", text)
    #  text = re.sub(r"&", " & ", text)
    #  text = re.sub(r"\|", " | ", text)
    #  text = re.sub(r";", " ; ", text)
    #  text = re.sub(r"\(", " ( ", text)
    #  text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text

def clear_character(sentence):
    pattern1= '\[.*?\]'
    pattern2 = re.compile('[^!^?^.^,^a-z^A-Z^0-9]')
    line1=re.sub(pattern1,' ',sentence)
    line2=re.sub(pattern2,' ',line1)
    new_sentence=' '.join(line2.split()) #去除空白
    return new_sentence


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        corpus = f.readlines()
    with open(sys.argv[2], 'w') as f:
        for i, dialog in enumerate(corpus):
            dialog = dialog.split('\t')
            #  print(dialog)
            src, tgt = clean_str(dialog[0]), clean_str(dialog[1])
            f.write(src)
            f.write('\t')
            f.write(tgt)
            f.write('\n')
            #  print(src, tgt)
            #  if i == 5:
            #      break


