def get_formatted_text(filename):
    
    '''
    Arguments: 
        filename: string, the name of the file to be read.
    Returns: 
        lines: list, a list of strings, each string is one line in the file.
    '''
    lines = []

    with open(filename, "r") as f:
        lines = f.readlines()
        lines = ["__" + line.strip().lower() + "__" for line in lines]

    return lines



def get_ngrams(line, n):
    
    '''
    Arguments: 
        line: string (a line of text), 
        n: int (the length of each n-gram)
    Returns: 
        ngrams: a list of n-grams
    Notes: (1) make sure to pad the beginning and end of the string with '_';
           (2) make sure to convert the string to lower-case, so "Hello" should be turned into "__hello__" before processing;
    '''
    ngrams = []  # init a list

    N = len(line)
    L = n
    for k in range(N - L + 1):
        ngrams.append(line[k : k + L])

    return ngrams

def get_dict(filename, n):
    
    '''
    Arguments: 
        filename: the filename to create an n-gram dictionary for, string
        n : Length of each n-gram, int
    Returns:
        ngram_dict: a dictionary, with ngrams as keys, and frequency of that ngram as the value,
            (the frequency is the number of times a key appears).
            
    Notes: Remember that get_formatted_text gives you a list of lines, and you want the ngrams from
        all the lines put together. (So please use get_formatted_text and get_ngrams)
    Hint: (1) dict.fromkeys(k, 0) will initialize a dictionary with the keys in k and an initial value of 0,
        k is an iterable specifying the keys of the new dictionary, it can be a list, tuple or a set;
        (2) assuming set1 is a set, set1.add(k) can add an element k into set1 only if k does not exist in set1 previously;
        (3) you can follow the step 1,2,3 to help you fill the 'get_dict' funtion if you want.
    
    '''
    # 1. get lines and combine them to a set of keys
    #   fill in

    # 2. use the set to initialize a dict
    #   fill in

    # 3. update the values of the dict
    #   fill in

    return ngram_dict

def top_N_common(filename, N, n):
    
    '''
    Arguments:
        filename: 
            string: the filename to generate a list of top N (most frequent n-gram, count) as key, value in the output dict
        N: 
            int: the number of most frequent n-gram to have in the output dict
        n : 
            int: Length of each n-gram
    Returns: 
        common_N: a dictionary representing the (n-gram, count) as (key, value) pairs that are most common in the file. 
                    It is highly recommended to sort by numerical value first, and for n-grams with the same count, 
                    sort them by name within the sorted dict.
    For example
    HINT:   (1) You may find the following StackOverflow post helpful for sorting a dictionary by its values: 
              https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    '''
    
    # 1. sort a dict
    # fill in

    # 2. get the top N common n-grams from the sorted dict and return it
    #   fill in


    return common_N


########################################## Checkpoint, can test code above before proceeding #################################################


def get_all_dicts(filename_list, n):
    
    '''
    Arguments: 
        filename_list: list (a list of filepaths for the different language text files to process). 
        n: int (the length of each n-gram)
    Returns: 
        lang_dicts: list, a list of dictionaries where there is a dictionary for each language file processed. Each dictionary in the list
                should have keys corresponding to the n-grams, and values corresponding to the count of the n-gram
    '''
    lang_dicts = []

    for lang in filename_list:
        langDict = get_dict(lang, n)
        lang_dicts.append(langDict)

    return lang_dicts

def dict_union(listOfDicts):
    
    '''
    Arguments:
        listOfDicts: list, A list of dictionaries where the keys are n-grams and the values are the count of the n-gram
    Returns:
        union_ngrams: list, An alphabetically sorted list containing all of the n-grams across all of the dictionaries in listOfDicts
    HINT:  (1) do not have duplicates n-grams)
           (2) It is recommended to use the "set" data type when doing this (look up "set union", or "set update" for python)
           (3) for alphabetically sorted, we mean that if you have a list of the n-grams altogether across all the languages,
              and you call sorted() on it, that is the output we want
           (4) you can follow the step1,2 to help you fill the 'dict_union' funtion if you want.
    '''
    # you can firstly initalize an empty set by: "union_ngrams = set()" and later convert it to a list

    # 1. update the set by using set.update()
    # fill in

    # 2. sort the set by converting it to a list and then sort
    # fill in

    return union_ngrams


def get_all_ngrams(langFiles, n):
    
    '''
    Arguments: 
        langFiles: list, a list of filepaths for the different language text files to process n. 
        n: int, the length of each n-gram
    Returns: 
        all_ngrams: list, a list of all the n-grams across the six languages
    '''
    all_ngrams = []
    all_ngrams = dict_union(get_all_dicts(langFiles, n))

    return all_ngrams


########################################## Checkpoint, can test code above before proceeding #############################################

def compare_langs(test_file, langFiles, N, n=3):
    
    '''
    Arguments:
        test_file: string,  mystery file's filepath to determine language of
        langFiles: list, list of filepaths of the languages to compare test_file to.
        N: int, the number of top n-grams for comparison
        n: int, length of n-gram, set to 3
    Returns:
        lang_match: string, the filepath of the language that has the highest number of top N matches that are similar to mystery file.
    HINT: (1) depending how you implemented top_N_common() earlier, you should only need to call it once per language,
        and doing so avoids a possible error
        (2) consider using the set method 'intersection()'
    '''
    # 1. get mystery top N using 'top_N_common' function
    # fill in

    # 2. cardinalities of intersections, use 'intersection()' and 'top_N_common' function
    # fill in

    return lang_match # it's a string


if __name__ == "__main__":
    from os import listdir
    from os.path import isfile, join, splitext

    # Testing top20Common()
    path = join("ngrams", "english.txt")
    print(top_N_common(path, 20, 4))

    # Compiling ngrams across all 7 languages (mystery is excluded) and finding the most similar language to mystery

    path = "ngrams"
    file_list = [f for f in listdir(path) if isfile(join(path, f))]
    file_list.remove("mystery.txt")
    path_list = [join(path, f) for f in file_list]

    print(get_all_ngrams(path_list, 4))  # list of all 4-grams spanning all languages

    # Find the similarity between languages all the languages and the mystery file
    test_file = join(path, "mystery.txt")
    print(compare_langs(test_file, path_list, 20))  # determine language of mystery file
