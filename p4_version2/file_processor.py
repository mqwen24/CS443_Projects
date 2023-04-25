'''imdb.py
Loads and preprocesses the IMDb dataset
YOUR NAMES HERE
CS443: Bio-Inspired Machine Learning
Project 2: Word Embeddings and Self-Organizing Maps (SOMs)
'''
import re
import numpy as np
import pandas as pd
import os

class FileProcessor():
    def read_files(self, folder_path, max_file=200):
        
        class_names = os.listdir(folder_path)
        num_classes = len(class_names)
        
        classes = []
        data = []
        
        file_num = 0
        
        # open class folder
        for i in range(num_classes):
            class_path = os.path.join(folder_path, class_names[i])
            
            file_names = os.listdir(class_path)
            # open text files
            for j in range(len(file_names)):
                # check if file is TXT format
                # print(file_names)
                if file_names[j][-4:] != ".txt":
                    continue
                
                
                file_path = os.path.join(class_path, file_names[j])
                file = open(file_path, "r", encoding="utf_8")

                # paragraphs = []
                for line in file:
                    if line.isspace():
                        continue
                    data.append(line)   
                    classes.append(i)
                    
                file_num = file_num + 1
                
                if file_num >= max_file:
                    break
                    
            # set file count to 0
            file_num = 0

        return data, classes
        
    def tokenize_words(self, text):
        '''Transforms a string sentence into words.

        Parameters:
        -----------
        text: string. Sentence of text.

        Returns:
        -----------
        list of strings. Words in the sentence `text`.

        This method is pre-filled for you (shouldn't require modification).
        '''
        # Define words as lowercase text with at least one alphabetic letter
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
        return pattern.findall(text.lower())


    def make_corpus(self, data, class_list, min_sent_size=5):
        '''Make the text corpus.
        Transforms text documents (list of strings) into a list of list of words (both Python lists).
        The format is [[<sentence>], [<sentence>], ...], where <sentence> = [<word>, <word>, ...].

        For the IMDb data, this transforms a list of reviews (each is a single string) into a list of
        sentences, where each sentence is represented as a list of string words. So the elements of the
        resulting list are the i-th sentence overall; we lose information about which review the
        sentence comes from.

        Parameters:
        -----------
        data: list of strings.
        min_sent_size: int. Don't add sentences LESS THAN this number of words to the corpus (skip over them).
            This is important because it removes empty sentences (bad parsing) and those with not enough
            word context.

        Returns:
        -----------
        corpus: list of lists (sentences) of strings (words in each sentence)

        TODO:
        - Split each review into sentences based on periods.
        - Tokenize the sentence into individual word strings (via tokenize_words())
        - Only add a list of words to the corpus if the length is at least `min_sent_size`.
        '''
        corpus = []

        classes = []
        for i in range(len(data)):
            paragraph = data[i]
            curr_class = class_list[i]
            sentences = paragraph.split(".")
            for sentence in sentences:
                words_list = self.tokenize_words(sentence)
                if len(words_list) >= min_sent_size:
                    corpus.append(words_list)
                    classes.append(curr_class)

        return corpus, classes


    def find_unique_words(self, corpus):
        '''Define the vocabulary in the corpus (unique words). Finds and returns a list of the unique words in the corpus.

        Parameters:
        -----------
        corpus: list of lists (sentences) of strings (words in each sentence).

        Returns:
        -----------
        unique_words: list of unique words in the corpus.
        '''
        visited = set()
        unique_words = []
        for sentence in corpus:
            for word in sentence:
                if word not in visited:
                    visited.add(word)
                    unique_words.append(word)

        return unique_words
    
    def determine_classes(self, corpus, unique_words, class_list, word2ind):
        '''Define the vocabulary in the corpus (unique words). Finds and returns a list of the unique words in the corpus.

        Parameters:
        -----------
        corpus: list of lists (sentences) of strings (words in each sentence).

        Returns:
        -----------
        unique_words: list of unique words in the corpus.
        '''
        unique_class = np.unique(np.array(class_list))
        num_words = len(unique_words)
        num_class = len(unique_class)
        
        appearance_matrix = np.zeros(shape=(num_words, num_class))
        
        for i in range(len(corpus)):
            curr_class = class_list[i]
            for word in corpus[i]:
                word_idx = word2ind[word]
                appearance_matrix[word_idx, curr_class] = appearance_matrix[word_idx, curr_class] + 1
        
        
        true_class = appearance_matrix.argmax(axis=1)

        return true_class


    def make_word2ind_mapping(self, vocab):
        '''Create dictionary that looks up a word index (int) by its string.
        Indices for each word are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: list of strings. Unique words in corpus.

        Returns:
        -----------
        Python dictionary with key,value pairs: string,int
        '''
        dict = {}
        for i in range(len(vocab)):
            dict[vocab[i]] = i

        return dict


    def make_ind2word_mapping(self, vocab):
        '''Create dictionary that uses a word int code to look up its word string
        Indices for each word are in the range [0, vocab_sz-1].

        Parameters:
        -----------
        vocab: list of strings. Unique words in corpus.

        Returns:
        -----------
        Python dictionary with key,value pairs: int,string
        '''
        dict = {}
        for i in range(len(vocab)):
            dict[i] = vocab[i]

        return dict


    def make_target_context_word_lists(self, corpus, word2ind, vocab_sz, context_win_sz=2):
        '''Make the target word array (training data) and context word array ("classes")

        Parameters:
        -----------
        corpus: list of lists (sentences) of strings (words in each sentence).
        word2ind: Dictionary mapping word string -> int code index. Range is [0, vocab_sz-1] inclusive.
        context_win_sz: int. How many words to include before/after the target word in sentences for context.

        Returns:
        -----------
        target_words_int: ndarray. shape=(N,) = (#target_words,)
            Each entry is the i-th int coded target word in corpus.
        context_words_int: ndarray of ndarrays. dtype=object.
            Each entry is an 1D ndarray containg the int codes for the context words associated with the i-th target word.
            outer shape: shape=(N,) = (#target_words,)
            shape(each inner ndarray) = (#context_words,).
            NOTE: #context_words is a variable value (NOT a constant!) in the range [context_win_sz, 2*context_win_sz].
            It is not always the same because of sentence boundary effects. This is why we're using a
            ndarray of ndarrays (not simply one multidimensional ndarray).
            NOTE: The context_words_int array needs to be created with dtype=object. This is because ndarrays need to have
            rectangular shapes — they can't be jagged (e.g. 1st row has length 10, 2nd row has length 11). Setting
            dtype=object allows the array to behave more like a Python list — it will assume you are storing a 1D array of
            objects of any kind (hence the dtype name) and not try to establish a consistent rectangular shape across the axes.
            In this case the "object" we are storing in the ndarray are other ndarray objects (each of which hold ints).

        HINT:
        - Search in a window `context_win_sz` words before after the current target in its sentence.
        Add int code indices of these context words to a ndarray and add this ndarray to the
        `context_words_int` list.
            - Only add context words if they are valid within the window. For example, only int codes of
            words on the right side of the first word of a sentence are added for that target word.


        Example:
        corpus = [['with', 'all', 'this', 'stuff', ...], ...]
        target_words_int  =   array([0, 1, 2, 3, ...])
        context_words_int =   array([array([1, 2]),
                                     array([0, 2, 3]),
                                     array([0, 1, 3, 4]),
                                     array([1, 2, 4, 5]),...])
        '''
        target_word_int = []
        context_words = []

        for sentence in corpus:
            for i in range(len(sentence)):
                target_word_int.append(word2ind[sentence[i]])

                # get a list of current context word
                curr_context_word = []
                for j in range(i-context_win_sz, i+context_win_sz+1):
                    if 0 <= j < len(sentence) and j != i:
                        curr_context_word.append(word2ind[sentence[j]])

                context_words.append(curr_context_word)

        # convert python list to ndarrays
        target_word_int = np.array(target_word_int)
        context_words_int = np.empty(len(target_word_int), dtype=object)
        for i in range(len(context_words)):
            context_words_int[i] = np.array(context_words[i])

        return target_word_int, context_words_int





    def prepare(self, folder_path):
        '''Preprocesses the raw IMDb dataset into sets of target and context words.

        This function ties together all the functions you wrote and tested in the notebook.
        For the most part, you should be able to copy-paste.

        Parameters:
        -----------
        path2imdb: str.
            Filename and relative path to the IMDb dataset .csv file.
            e.g. 'data/imdb_train.csv'
        num_reviews: int.
            Number of reviews to extract (starting from the beginning) from the IMDb dataset.

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

        TODO:
        - Import the dataset from disk
        - Select a subset of the reviews that go into the corpus
        - Make the corpus
        - Identify all the unique words in the corpus
        - Make word <-> int-code lookup table(s)
        - Collect int coded target words, int-coded context words
        '''
        data, classes = self.read_files(folder_path)
        
        corpus, sentence_classes = self.make_corpus(data, classes)
        unique_words = self.find_unique_words(corpus)
        vocab_sz = len(unique_words)
        word2ind = self.make_word2ind_mapping(unique_words)
        ind2word = self.make_ind2word_mapping(unique_words)
        
        word_classes = self.determine_classes(corpus, unique_words, sentence_classes, word2ind)

        targets_int, contexts_int = self.make_target_context_word_lists(corpus, word2ind, vocab_sz)
        return corpus, sentence_classes, targets_int, contexts_int, unique_words, word_classes, word2ind, ind2word

