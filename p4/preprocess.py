'''preprocess.py
Preprocess
Leo Qian, Muqing Wen
CS443: Bio-Inspired Machine Learning
Project 4
'''
import re
import os
from imdb import make_corpus, find_unique_words, make_word2ind_mapping, make_target_context_word_lists


def get_dataset(path2folder='data/enron/spam', num_emails=20):
    """
    This function will process emails in the folder indicated and return stuff necessary for the skipgram
    Parameters:
    -----------
    path2folder: str.
        Folder name to the txt files that you want to process
    num_emails: int.
        Number of emails to process from that folder

    Returns:
    -----------
    target_words_int: ndarray. shape=(N,) = (#target_words,)
        Each entry is the i-th int coded target word in corpus.
    context_words_int: ndarray of ndarrays. dtype=object.
        Each entry is an 1D ndarray containg the int codes for the context words associated with the i-th target word.
        outer shape: shape=(N,) = (#target_words,)
        shape(each inner ndarray) = (#context_words,).
    unique_words: List of unique words in the corpus.
    word2ind: Dictionary with key,value pairs: string,int mapping word to int-code.
    """

    # List all the files in the directory
    files = os.listdir(path2folder)

    # Load the contents of each file into a list of strings
    file_contents = []
    for file_name in files:
        file_path = os.path.join(path2folder, file_name)
        with open(file_path, "r") as file:
            file_contents.append(file.read())

    num_emails = min(len(file_contents), num_emails)

    email_dat = []  # initialize an empty list to store the extracted sentences

    for i in range(num_emails):  # iterate over each email in the dataset
        # find all sentences in the email based on punctuation marks using a regular expression
        sentences = re.findall(r"[\w\s]+[.!?;]+", file_contents[i])
        # remove all punctuation marks from each sentence using a regular expression
        sentences = [re.sub(r'[^\w\s]', '', s) for s in sentences]
        # remove leading/trailing whitespace and newline characters from each sentence
        sentences = [s.strip().replace('\n', '') for s in sentences]

        for sentence in sentences:  # iterate over each sentence in the email
            # add the sentence to the email_dat list
            email_dat.append(sentence)

    # build return variables

    # create a corpus (list of words) from the list of sentences
    corpus = make_corpus(email_dat)
    # find all unique words in the corpus
    unique_words = find_unique_words(corpus)
    # determine the size of the vocabulary (number of unique words)
    vocab_sz = len(unique_words)
    # create a mapping from each unique word to its index in the vocabulary
    word2ind = make_word2ind_mapping(unique_words)

    # create lists of target and context word indices for each word in the corpus
    targets_int, contexts_int = make_target_context_word_lists(corpus, word2ind, vocab_sz)
    return targets_int, contexts_int, unique_words, word2ind


if __name__ == "__main__":
    get_dataset('data/enron/spam', 100)
