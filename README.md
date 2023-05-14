# N gram distribution and TF-IDF analysis

We will do a basic textual analysis to study the
n-gram distribution of different languages, and examine a "mystery" text to
determine what language it is in.

We will then perform a TF-IDF analysis.
# Goals
We will:
* Write functions to perform a basic n-gram analysis of texts.
* Visualize the n-gram distribution of different texts.
* Use this information to analyze a new text.
* Compute TF-IDF scores for a set of documents, then find the most distinctive words.

# Background

## N-grams

n-grams are defined over either words in a sentence (the 3-grams in "words in a sentence" are "words in a" and "in a
sentence") or over the characters in a sentence (the 3-grams in "sentence"
are "sen" "ent" "nte" "ten" "enc" "nce"). In this homework, we will use
characters.

It is common to "pad" strings with dummy characters (e.g., `_`) to produce more
useful n-grams (to correctly capture, for example, that 's' is the most common
letter to start words through the 3-gram `__s`). The easiest way to do this is
to alter the sentence that you're building n-grams over by adding `_` (underscores) to the
beginning and end.

The interesting feature about n-grams is that different languages have different _distributions_ of n-grams. (In the simple case, the most common letter in English is `e`, but the most common letter in Spanish is `a`. The "1-grams" for English and Spanish have different distributions). This homework will build n-gram distributions for texts in different languages for various values of n, and you will find the pair of languages that are the most similar by analyzing their n-gram distributions. You will also see how the similarity value changes as we increase n.

## TF-IDF

 TF-IDF as a metric to find the "most distinctive" words in documents. However, TF-IDF can be used to analyze other parts of documents as well! In this homework, we will compute TF-IDF scores for a set of documents and use that to determine the most distinctive **n-gram** in each document. You can refer to the class notes on Brightspace for help with this part of the homework. You may use functions written in Problem 1 to complete this part.

## NLTK


We will use NLTK to clean the documents before processing the documents. To clean the documents, we will:

1) Remove *stop words* (words such as "the", "is", "and", etc.) from each document.
2) Remove punctuation from the document (you may use the `remove_punc` helper method in `helper.py` to help with this).
3) Make the words lower case.
4) Remove all spaces between the words.


### Step 1:
Make the functions `get_dict` and `top_N_common` as described in `hw8_1.py` to finally output a list of tuples containing the N most frequent (n-gram, count) pairs in decreasing order. `get_ngrams`, `get_dict`, and `top_N_common` also take in `n` as an argument which is the length of each n-gram. For example:
```
>>> top_N_common('ngrams/english.txt',20,4)
{' the': 126, 'the ': 119, 'and ': 106, ' and': 106, ' of ': 89, 'tion': 88, ' to ': 84, 'atio': 61, 'ion ': 58, 'righ': 54, 'ight': 54, ' rig': 54, 'one ': 43, ' in ': 42, 'all ': 39, 'e ri': 36, 'very': 35, 't to': 34, 's th': 34, 'ght ': 34}
```


### Step 2:


`get_all_dicts` takes in a list of filepath strings and returns a list of the n-gram dictionaries for all the files.
`get_all_ngrams` takes in a list of filepaths and returns one list of all n-grams across all the files.
`dict_union` takes in a list of dictionaries of n-grams for a group of files and creates one large alphabetically **sorted list** of all the n-grams across all the input dictionaries. Note that this larger output list doesn't involve the counts, it only involves the n-grams. It is highly recommended that you look into the "set" data type and methods such as `union` or `update`, as well as the dictionary method `keys`.

Our code can be tested w the following output:
```
[' a a', ' a b', ' a c', ' a d', ' a e', ' a f', ' a g', ' a h', ' a i', ' a l', ' a m', ' a n', ' a o', ' a p', ' a q', ' a r', ' a s', ' a t', ' a u', ' a v', ' a w', ' aba', ' abb', ' abe', ' acc', ' ace', ' ach', ' act', ' acu', ' ad ', ' ada', ' ade', ' adh', ' adm', ' adv', ' aff', ' afi', ' aga', ' age', ' agi', ' agr', ' aha', ' ahi', ' ai ', ' aid', ' aim', ' ain', ' aip', ' ait', ' aji', ' aju', ' aki', ' akt', ' al ', ' alc', ' alg', ' ali', ' all', ' alm', ' alo', ' als', ' alt', ' am ', ' ama', ' amb', ' ami', ' amm', ' amo', ' amp', ' amt', ' an ', ' ana', ' and', ' ane', ' anf', ' ang', ' ano', ' ans', ' ant', ' any', ' anz', ' ao ', ' aos', ' apl', ' app', ' aqu', ' arb', ' are', ' ari', ' arr', ' art', ' arz', ' as ', ' ase', ...
```
### Step 3:


`compare_langs` takes in a file that you want to determine the language of, a group of files of different languages for comparison, and finally,a value N that indicates how many of the top n-grams must be compared between the mystery file and    language file. What it should do is find the intersection of the top N n-grams between the mystery file and each language file, and determine which language file has the largest intersection. That language file's filepath should be the output.



## TF-IDF

For this problem, we will compute the tf-idf scores for the set of n-grams in each document in the `lecs/` folder.



<!-- Note that even after installing nltk and importing it, we need to add the following below your nltk import statement:
```
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
``` -->

## Step 1:
We have filled in the code for `read_and_clean_doc`. This function takes in a file path and outputs a cleaned string generated from the text in the file at the filepath. This function reads the file into a single string, removes punctuation (see `helper.py`), removes stopwords, and removes all whitespace from the string such that it is a sequence of only characters. This function returns the cleaned string.

From this we construct `build_doc_word_matrix`. This function takes in a list of filepaths and a number `n` equal to the number of characters in an ngram. You should generate a document word matrix with columns representing ngrams and rows representing each document in the list given. You may use functions previously written in `hw8_1.py` to complete this part, if you wish.

## Step 2:
Construct `build_tf_matrix` and `build_idf_matrix`. In addition to the instructions given in `hw8_2.py`, you may want to refer to the lecture notes on brightspace for how to complete these functions.

Fill in the code for `build_tfidf_matrix`.

## Step 3:
Fill in the code for `find_distinctive_ngrams`. This function calculates and outputs the top 3 most unique words found in each document (Note: the most unique words, not the most common). This function takes in a doc-word matrix, a list of ngrams, and a list of filepaths. You should return a dictionary with the following qualities: each filepath becomes a key in the dictionary. The value for each filepath key should be a list of the top 4 most unique ngrams in the file. 

While we will test you using our own tests, at the end of `hw8_2.py` are some good test cases to check your code as you go. 


*** Testing build_doc_word_matrix ***
(2, 3413)
3413
[ 1.  1.  2.  2.  1. 16.  0.  0.  1.  1.]
['0038', '0098', '00mb', '00me', '0138', '0211', '02co', '02fa', '02pa', '0380']
[0. 0. 0. 0. 0. 2. 1. 1. 2. 0.]

*** Testing build_tf_matrix ***
[0.00024795 0.00024795 0.00049591 0.00049591 0.00024795 0.00396727
 0.         0.         0.00024795 0.00024795]
[0.         0.         0.         0.         0.         0.00078401
 0.000392   0.000392   0.00078401 0.        ]
[1. 1.]

*** Testing build_idf_matrix ***
[0.30103 0.30103 0.30103 0.30103 0.30103 0.      0.30103 0.30103 0.
 0.30103]

*** Testing build_tfidf_matrix ***
(2, 3413)
[7.46417049e-05 7.46417049e-05 1.49283410e-04 1.49283410e-04
 7.46417049e-05 0.00000000e+00 0.00000000e+00 0.00000000e+00
 0.00000000e+00 7.46417049e-05]
[0.       0.       0.       0.       0.       0.       0.000118 0.000118
 0.       0.      ]

*** Testing find_distinctive_words ***
{'lecs/1_vidText.txt': ['econ', 'seco', 'rsec'], 'lecs/2_vidText.txt': ['uter', 'sspo', 'oute']}
```
