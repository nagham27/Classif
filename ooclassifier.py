import sys
import copy

Debug = False  # for debugging
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # input from: Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # input from: file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)


class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0 # TP= True positives
        self.FP = 0 # FP= False positives
        self.TN = 0 # TN= True negatives
        self.FN = 0 # FN= False negatives
        return

    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    def set_target_words(self, lw):
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords Hardcoded (%d): " % ln, end='')
        print(self.get_target_words())
        return

    def print_run_info(self):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)

    def classify_all(self, ts, update=True, tlabel="classify_all"):
        for ti in ts.get_instances():
            cl, e = self.classify(ti, update=update, tlabel=tlabel)
        return


class TrainingInstance(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inst = dict()
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=True
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)

    def remove_punctuation(self, wordList):
        """
        Takes in a list of words and removes
        any punctuation present in the words,
        returns a string of space separated words.

        Argumets:
            wordList (list):A list of words.

        Returns:
            final (string):A string of space separated words
            free of punctuation.
        """

        word = ""
        newlist = []
        # for every word in wordLlist
        for i in wordList:
            for j in list(i):
                # check if character is a number or letter
                if j.isdigit() or j.isalpha():
                    # concatenate letters/numbers
                    word += j
            # append word without punctuation to list
            newlist.append(word)
            word = ""
        final = " ".join(newlist)
        # function returns final
        return(final)

    def remove_numbers(self, wordList):
        """
        Takes in a list of words and removes
        any numbers present in the words,unless the
        token consists only of numbers,
        returns a string of space separated words.

        Argumets:
            wordList (list):A list of words.

        Returns:
            final (string):A string of space separated words
            free of numbers,unless token is a number.
        """
        word = ""
        newlist = []
        # for every word in wordList
        for i in wordList:
            # if token consists of only numbers
            if i.isdigit():
                word = i
                newlist.append(word)
                word = ""
            # if token consists of more than just numbers
            else:
                for j in list(i):
                    # check if character is not a number
                    if not j.isdigit():
                        # concatenate letters/punctuation
                        word += j
                newlist.append(word)
                word = ""
        final = " ".join(newlist)
        # function returns final
        return(final)

    def remove_stopwords(self, wordList):

        """
        Takes in a list of words and removes
        any stop words present,returns a string
        of space separated words.

        Argumets:
            wordList (list):A list of words.

        Returns:
            final (string):A string of space separated words
            free of stop words.
        """
        # list of stop words
        stop_words = ["i", "me", "my", "myself", "we", "our",
                      "ours", "ourselves", "you", "your",
                      "yours", "yourself", "yourselves", "he",
                      "him", "his", "himself", "she", "her",
                      "hers", "herself", "it", "its", "itself",
                      "they", "them", "their", "theirs",
                      "themselves", "what", "which", "who", "whom",
                      "this", "that", "these", "those",
                      "am", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had",
                      "having", "do", "does", "did", "doing", "a",
                      "an", "the", "and", "but", "if",
                      "or", "because", "as", "until", "while",
                      "of", "at", "by", "for", "with",
                      "about", "against", "between", "into",
                      "through", "during", "before", "after",
                      "above", "below", "to", "from", "up", "down",
                      "in", "out", "on", "off", "over", "under", "again",
                      "further", "then", "once", "here", "there", "when",
                      "where", "why", "how", "all", "any", "both", "each",
                      "few", "more", "most", "other", "some", "such", "no",
                      "nor", "not", "only", "own", "same", "so", "than",
                      "too", "very", "s", "t", "can", "will", "just",
                      "don", "should", "now"]
        newlist = []
        # for every word in wordList
        for i in wordList:
            # if word is not in stop_words
            if i not in stop_words:
                newlist.append(i)
        final = " ".join(newlist)
        # function returns final
        return(final)

    def lowerCase(self, wordList):
        newlist = []
        # for every word in wordList
        for i in wordList:
            # convert all letters to lowercase letters
            newlist.append(i.lower())
        return newlist

    def preprocess_words(self, mode=''):

        if mode == '':
            # convert all letters to lowercase letters
            all_lower = self.lowerCase(self.inst["words"])
            # remove punctuation from words
            no_punctuation = self.remove_punctuation(all_lower)
            # remove numbers from words
            no_numbers = self.remove_numbers(no_punctuation.split())
            # remove stop words
            no_stopwords = self.remove_stopwords(no_numbers.split())
            self.inst["words"] = no_stopwords.split()
            return
        else:
            if mode == "keep-digits":
                # convert all letters to lowercase letters
                all_lower = self.lowerCase(self.inst["words"])
                # remove punctuation from words
                no_punctuation = self.remove_punctuation(all_lower)
                # remove stop words
                no_stopwords = self.remove_stopwords(no_punctuation.split())
                self.inst["words"] = no_stopwords.split()
                return
            elif mode == "keep-stops":
                # convert all letters to lowercase letters
                all_lower = self.lowerCase(self.inst["words"])
                # remove punctuation from words
                no_punctuation = self.remove_punctuation(all_lower)
                # remove numbers from words
                no_numbers = self.remove_numbers(no_punctuation.split())
                self.inst["words"] = no_numbers.split()
                return
            elif mode == "keep-symbols":
                # convert all letters to lowercase letters
                all_lower = self.lowerCase(self.inst["words"])
                # remove numbers from words
                no_numbers = self.remove_numbers(all_lower)
                # remove stop words
                no_stopwords = self.remove_stopwords(no_numbers.split())
                self.inst["words"] = no_stopwords.split()
                return


class TrainingSet(C274):
    def __init__(self):
        self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary
        self.variable = dict()
        return

    def get_instances(self):
        return(self.inObjHash)

    def get_lines(self):
        return(self.inObjList)

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return
    
    # classify all training instances
    def classify_all(self, run, update=True, tlabel="classify_all"):
        for ti in self.get_instances():
            cl, e = run.classify(ti, update=update, tlabel=tlabel)
        return

    def set_env_variable(self, k, v):
        self.variable[k] = v
        return

    def get_env_variable(self, k):
        if k in self.variable:
            return(self.variable[k])
        else:
            return ""

    def inspect_comment(self, line):
        if len(line) > 1 and line[1] != ' ':
            v = line.split(maxsplit=1)
            self.set_env_variable(v[0][1:], v[1])
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            # Check for comments
            if line[0] == '%':  # Comments must start with %
                self.inspect_comment(line)
                continue

            # Save the training data input, before parsing
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    def preprocess(self, mode=''):
        # for every training instance in the trainining set
        for ti in self.inObjHash:
            # preprocess the training instance
            ti.preprocess_words(mode=mode)
        return

    def return_nfolds(self, num=3):
        listOfObj = []
        for i in range(num):
            # create new object of class TrainingSet
            newObj = TrainingSet()
            listOfObj.append(newObj)
            # create folds using a round robin method
        for i in range(len(self.inObjHash)):
            # deep copy all attributes of the original Training set
            # into the new objects
            listOfObj[i % num].inObjHash.append(copy.deepcopy(self.inObjHash[i]))
            listOfObj[i % num].inObjList.append(copy.deepcopy(self.inObjList[i]))
            listOfObj[i % num].type = copy.deepcopy(self.type)
            listOfObj[i % num].variable = copy.deepcopy(self.variable)
        return listOfObj

    def copy(self):
        # create a new object of class TrainingSet
        x = TrainingSet()
        # deep copy all attributes of the original Training set
        # into the new object
        x.inObjHash = copy.deepcopy(self.inObjHash)
        x.inObjList = copy.deepcopy(self.inObjList)
        x.type = copy.deepcopy(self.type)
        x.variable = copy.deepcopy(self.variable)
        return x

    def add_fold(self, tset):
        # for every training instance in tset
        for ti in tset.get_instances():
            # add the training instace to an object of
            # class TrainingSet
            self.inObjHash.append(copy.deepcopy(ti))
        # for every unparsed line in tset
        for i in tset.get_lines():
            # add the unparsed line to an object of
            # class TrainingSet
            self.inObjList.append(copy.deepcopy(i))
        return


class ClassifyByTopN(ClassifyByTarget):

    def target_top_n(self, tset, num=5, label=''):
        myWords = []
        uniqueWords = []
        freqCount = []
        myTargetWords = []
        # for every training instance in tset
        for ti in tset.get_instances():
            # check if labels match
            if ti.get_label() == label:
                for w in ti.get_words():
                    if w[0] == '#':
                        pass
                    else:
                        # form a list containing
                        # all words in training instances
                        myWords.append(w)

        for word in myWords:
            if word not in uniqueWords:
                # form a list of unique words
                uniqueWords.append(word)
        for word in uniqueWords:
            # count the frequency of the words
            # in the original list
            x = myWords.count(word)
            freqCount.append(x)
        # sort list
        sortedFreq = sorted(freqCount)
        # sort from biggest to smallest number
        sortFreq = list(reversed(sortedFreq))
        for i in range(num):
            for j in range(len(freqCount)):
                # check for words that have the same frequency
                if sortFreq[i] == freqCount[j]:
                    # check if word has not been appended already
                    if uniqueWords[j] not in myTargetWords:
                        # form a list containing the top num
                        # most frequent words
                        myTargetWords.append(uniqueWords[j])
        # set target words to be myTargetWords
        self.set_target_words(myTargetWords)
        return


def basemain():
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    print(run1)     #show __str__
    lr = [run1]
    print(lr)       #show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, '#weather')

    return


def base1main():
    tset = TrainingSet()
    run1 = ClassifyByTarget()

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    print("********************************************")
    pfeatures = tset.get_env_variable("pos-features")
    print("pos-features: ", pfeatures)
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("********************************************")


    run1.set_target_words(pfeatures.strip().split())
    run1.classify_all(tset)
    run1.print_config()
    run1.eval_training_set(tset, plabel)

    tp, fp, tn, fn = run1.get_TF()
    precision = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    accuracy = float(tp + tn) / float(tp + tn + fp + fn)
    print("Accuracy:  %3.2g = " % accuracy, end='')
    print("(%d + %d) / (%d + %d + %d + %d)" % (tp, tn, tp, tn, fp, fn))
    print("Precision: %3.2g = " % precision, end='')
    print("%d / (%d + %d)" % (tp, tp, fp))
    print("Recall:    %3.2g = " % recall, end='')
    print("%d / (%d + %d)" % (tp, tp, fn))
    return


if __name__ == "__main__":
    base1main()
