#!/usr/bin/env python
from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
import codecs
import sys
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer

""" List of Stopwords : 318 """
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS 
#from nltk.corpus import stopwords #NLTK stop words - 153

""" Punctuations """
from nltk.tokenize import RegexpTokenizer

""" Bigram """
from nltk import ngrams



kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        None

    def features(self, text):
        d = defaultdict(int)


# Length of every word - decreasing the accuracy, above average is performing better        
        """for ii in text :
            d[len(ii)]+=1  """
                  
# Split to bigram, trigram and fourgram - Bigram is decreasing the performance
        """token = text.split()
        for ngram in ngrams(token, 2):
            d[ngram] +=1"""
        
        token = text.split()
        for trigram in ngrams(token, 3):
            d[trigram] +=1
         
        for fourgram in ngrams(token, 4):
            d[fourgram] +=1

# Length of the text
        text_split= text.split()
        sent_len=len(text_split)
        d['text_length'] = sent_len
 
# Length of the every word is calculated and the average is calculated by the (length of the word) / (length of the sentence)
        wordlen = 0
        for ii in text_split :
            wordlen += len(ii)      
        d['wordlen'] = wordlen 
        d['avg_len'] = wordlen/sent_len

#POS Tagging and the last tag for every sentence is added to Dictionary            
        tags = []
        for word,tag in nltk.pos_tag(text):
            tags.append(tag)
        for last_tag in tag:
            d[last_tag] += 1 
            
        last_two_pos =  tags[-2] + tags[-1] 
        d[last_two_pos] += 1


#Remove Punctuations
        tokenizer = RegexpTokenizer(r'\w+') #Punctuations
#Tokenize text and remove stop words from sklearn (More stop words than in NLTK ie., 318)
        tokenized_word = tokenizer.tokenize(text) #tokenize the text 

#scikit-learn stopwords      
        filter_stopword = []
        for word in tokenized_word :
            if not word in ENGLISH_STOP_WORDS :
                filter_stopword.append(word)

#NLTK stopwords
        """ENGLISH_STOP_WORDS = stopwords.words('english')
        filter_stopword = []
        for word in tokenized_word :
            if not word in ENGLISH_STOP_WORDS :
                filter_stopword.append(word)"""

        wordsfiltered = filter_stopword #After the stopwords are removed
        
        for ii in wordsfiltered :
            d[morphy_stem(ii)] += 1  
        return d
       
reader = codecs.getreader('utf8')
writer = codecs.getwriter('utf8')


def prepfile(fh, code):
  if type(fh) is str:
    fh = open(fh, code)
  ret = gzip.open(fh.name, code if code.endswith("t") else code+"t") if fh.name.endswith(".gz") else fh
  if sys.version_info[0] == 2:
    if code.startswith('r'):
      ret = reader(fh)
    elif code.startswith('w'):
      ret = writer(fh)
    else:
      sys.stderr.write("I didn't understand code "+code+"\n")
      sys.exit(1)
  return ret

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--trainfile", "-i", nargs='?', type=argparse.FileType('r'), default=sys.stdin, help="input train file")
    parser.add_argument("--testfile", "-t", nargs='?', type=argparse.FileType('r'), default=None, help="input test file")
    parser.add_argument("--outfile", "-o", nargs='?', type=argparse.FileType('w'), default=sys.stdout, help="output file")
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this fraction of total')
    args = parser.parse_args()
    trainfile = prepfile(args.trainfile, 'r')
    if args.testfile is not None:
        testfile = prepfile(args.testfile, 'r')
    else:
        testfile = None
    outfile = prepfile(args.outfile, 'w')

    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(trainfile, delimiter='\t')
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    sys.stderr.write("Training classifier ...\n")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    sys.stderr.write("Accuracy on dev: %f\n" % (float(right) / float(total)))

    if testfile is None:
        sys.stderr.write("No test file passed; stopping.\n")
    else:
        # Retrain on all data
        classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)

        # Read in test section
        test = {}
        for ii in DictReader(testfile, delimiter='\t'):
            test[ii['id']] = classifier.classify(fe.features(ii['text']))

        # Write predictions
        o = DictWriter(outfile, ['id', 'pred'])
        o.writeheader()
        for ii in sorted(test):
            o.writerow({'id': ii, 'pred': test[ii]})