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

    email_dat = []
    for i in range(num_emails):
        # split the email by punctuation marks using regular expressions and remove punctuation
        sentences = re.findall(r"[\w\s]+[.!?;]+", file_contents[i])
        sentences = [re.sub(r'[^\w\s]', '', s) for s in sentences]
        sentences = [s.strip().replace('\n', '') for s in sentences]

        for sentence in sentences:
            email_dat.append(sentence)

    # build return variables
    corpus = make_corpus(email_dat)
    unique_words = find_unique_words(corpus)
    vocab_sz = len(unique_words)
    word2ind = make_word2ind_mapping(unique_words)

    targets_int, contexts_int = make_target_context_word_lists(corpus, word2ind, vocab_sz)
    return targets_int, contexts_int, unique_words, word2ind


if __name__ == "__main__":
    get_dataset('data/enron/spam', 100)
