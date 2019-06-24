import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *


from textblob import TextBlob
import spacy

#nlp = spacy.load('en')

#The Disappearing Parachutist


# Structure and Layout
window = Tk()
window.title("IR")
window.geometry("700x400")
window.config(background='black')

# TAB LAYOUT
tab_control = ttk.Notebook(window)

tab1 = ttk.Frame(tab_control)
tab2 = ttk.Frame(tab_control)
tab3 = ttk.Frame(tab_control)

# ADD TABS TO NOTEBOOK
tab_control.add(tab1, text='URL')
tab_control.add(tab2, text='File Processer')
tab_control.add(tab3, text='About')

label1 = Label(tab1, text='INFORMATION RETREIVAL PROJECT', padx=5, pady=5)
label1.grid(column=0, row=0)

label2 = Label(tab2, text='File Processing', padx=5, pady=5)
label2.grid(column=0, row=0)

label3 = Label(tab3, text='About', padx=5, pady=5)
label3.grid(column=0, row=0)

tab_control.pack(expand=1, fill='both')

about_label = Label(tab3, text="INFORMATION RETREIVAL PROJECT \n PREDICTING ARTICLE FOR FROM CRIMINAL CASES", pady=5, padx=5)
about_label.grid(column=0, row=1)

summary=""
# Functions FOR NLP  FOR TAB ONE
def get_tokens():   #summarize
    raw_text = str(raw_entry.get())
    new_text = TextBlob(raw_text)
    import bs4 as bs
    import urllib.request
    import re
    import nltk
    import heapq

    scraped_data = urllib.request.urlopen(raw_text)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    # preprocessing
    # removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    # weighted frequency of occurance
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # print(word_frequencies)

    # calculating sentence scores

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # print(sentence_scores)

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)

    tab1_display.insert(tk.END, summary)


def get_pos_tags():
    raw_text = str(raw_entry.get())
    new_text = TextBlob(raw_text)

    import bs4 as bs
    import urllib.request
    import re
    import nltk
    import heapq

    scraped_data = urllib.request.urlopen(raw_text)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    # preprocessing
    # removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    # weighted frequency of occurance
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # print(word_frequencies)

    # calculating sentence scores

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # print(sentence_scores)

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)

    tab1_display.insert(tk.END, summary)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    import operator

    d = []

    def preprocess(input_str):
        import re
        from nltk.tokenize import sent_tokenize, word_tokenize
        import string
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize

        input_str = input_str.lower()
        result = re.sub(r'\d+', '', input_str)
        input_str = input_str.translate(str.maketrans("", "", string.punctuation))
        input_str = str(input_str.strip())
        input_str = word_tokenize(input_str)
        stop_words = set(stopwords.words('english'))
        input_str = [i for i in input_str if not i in stop_words]

        s = ""
        stemmer = PorterStemmer()
        for word in input_str:
            s += (stemmer.stem(word) + " ")
        return s

    with open("doc1.txt", 'r') as myfile:
        doc1 = myfile.read()
        doc1 = preprocess(doc1)
    d.append(doc1)

    T = 1

    relevant = [d[0]]
    rel = d[0]
    words = []
    s = ""
    # q = input(summary)
    for i in range(T):
        s += d[i]
    words.append(list(s.split()))
    words1 = list(set(list(s.split())))

    documents = words1
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    print(X)

    true_k = 20
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    tab1_display.insert(tk.END, "\n\nCLUSTERS\n")

    for i in range(true_k):
        print("Cluster %d:" % i),

        tab1_display.insert(tk.END, "CLUSTER " + str(i) + " : ")
        for ind in order_centroids[i, :10]:

            tab1_display.insert(tk.END,terms[ind])
            tab1_display.insert(tk.END, " ")
        tab1_display.insert(tk.END, "\n")

    print("\n")
    print("Prediction")

    Y = vectorizer.transform([summary])
    prediction = model.predict(Y)
    tab1_display.insert(tk.END,"PREDICTED CLUSTER : ")
    tab1_display.insert(tk.END, prediction)




def get_pos_tags1():
    raw_text = str(raw_entry.get())
    new_text = TextBlob(raw_text)

    import bs4 as bs
    import urllib.request
    import re
    import nltk
    import heapq

    scraped_data = urllib.request.urlopen(raw_text)
    article = scraped_data.read()

    parsed_article = bs.BeautifulSoup(article, 'lxml')

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    # preprocessing
    # removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    # weighted frequency of occurance
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # print(word_frequencies)

    # calculating sentence scores

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # print(sentence_scores)

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)

    tab1_display.insert(tk.END, summary)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    import operator

    d = []

    def preprocess(input_str):
        import re
        from nltk.tokenize import sent_tokenize, word_tokenize
        import string
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize

        input_str = input_str.lower()
        result = re.sub(r'\d+', '', input_str)
        input_str = input_str.translate(str.maketrans("", "", string.punctuation))
        input_str = str(input_str.strip())
        input_str = word_tokenize(input_str)
        stop_words = set(stopwords.words('english'))
        input_str = [i for i in input_str if not i in stop_words]

        s = ""
        stemmer = PorterStemmer()
        for word in input_str:
            s += (stemmer.stem(word) + " ")
        return s

    with open("doc1.txt", 'r') as myfile:
        doc1 = myfile.read()
        doc1 = preprocess(doc1)
    d.append(doc1)

    T = 1

    relevant = [d[0]]
    rel = d[0]
    words = []
    s = ""
    # q = input(summary)
    for i in range(T):
        s += d[i]
    words.append(list(s.split()))
    words1 = list(set(list(s.split())))

    documents = words1
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    print(X)

    true_k = 20
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    tab1_display.insert(tk.END, "\n\nCLUSTERS\n")
    for i in range(true_k):
        print("Cluster %d:" % i),

        tab1_display.insert(tk.END, "CLUSTER " + str(i) + " : ")
        for ind in order_centroids[i, :10]:
            tab1_display.insert(tk.END, terms[ind])
            tab1_display.insert(tk.END, " ")
        tab1_display.insert(tk.END, "\n")

    print("\n")
    print("Prediction")

    Y = vectorizer.transform([summary])
    prediction = model.predict(Y)
    tab1_display.insert(tk.END, "PREDICTED CLUSTER : ")
    tab1_display.insert(tk.END, prediction)
    #https://en.wikipedia.org/wiki/Punjab_National_Bank_Scam
    #predicting the article number
    query_no=prediction
    final_list=[]
    for i in range(true_k):
        if(i==query_no):

            for ind in order_centroids[i, :]:
                final_list.append(terms[ind])
    print(final_list)

    d=[]
    with open("doc2.txt", 'r') as myfile:
        doc2 = myfile.read()
        doc2 = preprocess(doc2)

    d.append(doc2)
   # print(d)

    with open("doc3.txt", 'r') as myfile:
        doc3 = myfile.read()
        doc3 = preprocess(doc3)  # type: str
    d.append(doc3)

    with open("doc4.txt", 'r') as myfile:
        doc4 = myfile.read()  # type: str
        doc4 = preprocess(doc4)
    d.append(doc4)

    with open("doc5.txt", 'r') as myfile:
        doc5 = myfile.read()
        doc5 = preprocess(doc5)

    d.append(doc5)
    with open("doc6.txt", 'r') as myfile:
        doc6 = myfile.read()
        doc6 = preprocess(doc6)
    d.append(doc6)

    with open("doc7.txt", 'r') as myfile:
        doc7 = myfile.read()  # type: str
        doc7 = preprocess(doc7)
    d.append(doc7)

    with open("doc8.txt", 'r') as myfile:
        doc8 = myfile.read()  # type: str
        doc8 = preprocess(doc8)
    d.append(doc8)

    with open("doc9.txt", 'r') as myfile:
        doc9 = myfile.read()  # type: str
        doc9 = preprocess(doc9)
    d.append(doc9)
    with open("doc10.txt", 'r') as myfile:
        doc10 = myfile.read()  # type: str
        doc10 = preprocess(doc10)
    d.append(doc10)
    T = 9

    relevant = [ d[0],d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8]]
    rel = d[0]+d[1] + d[2] + d[3] + d[4] + d[5] + d[6] + d[7] + d[8]
    words = []
    s = ""
    q = " ".join(final_list)
    print(q)
    for i in range(T):
        s += d[i]
    s += q
    words.append(list(s.split()))
    words1 = list(set(list(s.split())))

    N1_W = []
    for i in range(len(words1)):
        count = 0
        for j in range(len(relevant)):
            if words1[i] in relevant[j]:
                count += 1
        N1_W.append([(words1[i]), count, ((T - count + 0.5) / (count + 0.5))])
    print(N1_W)

    final = {}
    for i in range(len(d)):
        d1 = set(d[i].split())
        d2 = set(q.split())
        d3 = list(d1 & d2)
        res = 1
        for j in range(len(d3)):
            for k in range(len(N1_W)):
                if ((d3[j]) == N1_W[k][0]):
                    res *= N1_W[k][2]
        final[i] = res
        print(res)

    sorted_x = sorted(final.items(), key=operator.itemgetter(1), reverse=True)
    #final list
    ll=sorted_x
    tab1_display.insert(tk.END,"\n")
    for i in range(len(ll)):
        a=ll[i][0]
        print("article no: "+str(a))
        if a==0:
            print("STATES AND UNION TERRETORIES")
            tab1_display.insert(tk.END, "ARTICLE 1 AND 2")
            tab1_display.insert(tk.END, "\n")
        if a==1:
            print("CITIZENSHIP")
            tab1_display.insert(tk.END, "CITIZENSHIP")
            tab1_display.insert(tk.END, "\n")
        if a==2:
            print("FUNDAMENTAL RIGHTS")
            tab1_display.insert(tk.END, "FUNDAMENTAL RIGHTS")
            tab1_display.insert(tk.END, "\n")

        if a==3:
            print("DIRECTIVE PRINCIPLES OF STATE POLICY AND FUNDAMENTAL DUTIES")
            tab1_display.insert(tk.END, " DIRECTIVE PRINCIPLES OF STATE POLICY AND FUNDAMENTAL DUTIES")
            tab1_display.insert(tk.END, "\n")
        if a==4:
            print("THE UNION")
            tab1_display.insert(tk.END, "THE UNION")
            tab1_display.insert(tk.END, "\n")
        if a==5:
            print("THE STATES")
            tab1_display.insert(tk.END, "THE STATES")
            tab1_display.insert(tk.END, "\n")

        if a==6:
            print("THE UNION TERRETORIES")
            tab1_display.insert(tk.END, "THE UNION TERRETORIES")
            tab1_display.insert(tk.END, "\n")
        if a==7:
            print("THE MUNICIPALITIES")
            tab1_display.insert(tk.END, "THE MUNICIPALITIES")
            tab1_display.insert(tk.END, "\n")
        if a==8:
            print("THE SCHEDULED AND TRIBAL AREAS")
            tab1_display.insert(tk.END,"THE SCHEDULED AND TRIBAL AREAS" )
            tab1_display.insert(tk.END, "\n")


        tab1_display.insert(tk.END, a)
        tab1_display.insert(tk.END,"\n")
    print(sorted_x)


# Clear entry widget
def clear_entry_text():
    entry1.delete(0, END)


def clear_display_result():
    tab1_display.delete('1.0', END)


# Clear Text  with position 1.0
def clear_text_file():
    displayed_file.delete('1.0', END)


# Clear Result of Functions
def clear_result():
    tab2_display_text.delete('1.0', END)


# Functions for TAB 2 FILE PROCESSER
# Open File to Read and Process
def openfiles():
    file1 = tk.filedialog.askopenfilename(filetypes=(("Text Files", ".txt"), ("All files", "*")))
    read_text = open(file1).read()
    displayed_file.insert(tk.END, read_text)


def get_file_tokens():
    raw_text = displayed_file.get('1.0', tk.END)
    new_text = TextBlob(raw_text)
    import bs4 as bs
    import urllib.request
    import re
    import nltk
    import heapq

    scraped_data = raw_text




    article_text = scraped_data

    # preprocessing
    # removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    # weighted frequency of occurance
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # print(word_frequencies)

    # calculating sentence scores

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # print(sentence_scores)

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)

    tab2_display_text.insert(tk.END, summary)



def get_file_pos_tags():
    raw_text = displayed_file.get('1.0', tk.END)
    new_text = TextBlob(raw_text)
    import bs4 as bs
    import urllib.request
    import re
    import nltk
    import heapq

    scraped_data = raw_text



    article_text = raw_text
    # preprocessing
    # removing square brackets and extra spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    # converting text to sentences
    sentence_list = nltk.sent_tokenize(article_text)

    # print(sentence_list)

    # weighted frequency of occurance
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    # print(word_frequencies)

    # calculating sentence scores

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    # print(sentence_scores)

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)

    tab1_display.insert(tk.END, summary)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    import operator

    d = []

    def preprocess(input_str):
        import re
        from nltk.tokenize import sent_tokenize, word_tokenize
        import string
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize

        input_str = input_str.lower()
        result = re.sub(r'\d+', '', input_str)
        input_str = input_str.translate(str.maketrans("", "", string.punctuation))
        input_str = str(input_str.strip())
        input_str = word_tokenize(input_str)
        stop_words = set(stopwords.words('english'))
        input_str = [i for i in input_str if not i in stop_words]

        s = ""
        stemmer = PorterStemmer()
        for word in input_str:
            s += (stemmer.stem(word) + " ")
        return s

    with open("doc1.txt", 'r') as myfile:
        doc1 = myfile.read()
        doc1 = preprocess(doc1)
    d.append(doc1)

    T = 1

    relevant = [d[0]]
    rel = d[0]
    words = []
    s = ""
    # q = input(summary)
    for i in range(T):
        s += d[i]
    words.append(list(s.split()))
    words1 = list(set(list(s.split())))

    documents = words1
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    print(X)

    true_k = 5
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    tab2_display_text.insert(tk.END, "\n\nCLUSTERS\n")
    for i in range(true_k):
        print("Cluster %d:" % i),

        tab2_display_text.insert(tk.END, "CLUSTER " + str(i) + " : ")
        for ind in order_centroids[i, :10]:
            tab2_display_text.insert(tk.END, terms[ind])
            tab2_display_text.insert(tk.END, " ")
        tab2_display_text.insert(tk.END, "\n")

    print("\n")
    print("Prediction")

    Y = vectorizer.transform([summary])
    prediction = model.predict(Y)
    tab2_display_text.insert(tk.END, "PREDICTED CLUSTER : ")
    tab2_display_text.insert(tk.END, prediction)




def get_file_sentiment():
    raw_text = displayed_file.get('1.0', tk.END)
    new_text = TextBlob(raw_text)
    final_text = new_text.sentiment
    result = '\nSubjectivity:{}, Polarity:{}'.format(new_text.sentiment.subjectivity, new_text.sentiment.polarity)
    tab2_display_text.insert(tk.END, result)


def get_file_entities():
    raw_text = displayed_file.get('1.0', tk.END)
    docx = nlp(raw_text)
    final_text = [(entity.text, entity.label_) for entity in docx.ents]
    result = '\nEntities:{}'.format(final_text)
    tab2_display_text.insert(tk.END, result)


def nlpiffy_file():
    raw_text = displayed_file.get('1.0', tk.END)
    docx = nlp(raw_text)
    final_text = [(token.text, token.shape_, token.lemma_, token.pos_) for token in docx]
    result = '\nSummary:{}'.format(final_text)
    tab2_display_text.insert(tk.END, result)


# MAIN NLP TAB
l1 = Label(tab1, text="Enter URL For Analysis")
l1.grid(row=1, column=0)

raw_entry = StringVar()
entry1 = Entry(tab1, textvariable=raw_entry, width=50)
entry1.grid(row=1, column=1)

# bUTTONS
button1 = Button(tab1, text="Summarize", width=12, command=get_tokens, bg='#03A9F4', fg='#fff')
button1.grid(row=4, column=0, padx=10, pady=10)

button2 = Button(tab1, text="PREDICT", width=12, command=get_pos_tags, bg='#BB86FC')
button2.grid(row=4, column=1, padx=10, pady=10)

button3 = Button(tab1, text="FINAL PREDICT", width=12, command=get_pos_tags1, bg='#BB86FC')
button3.grid(row=4, column=2, padx=10, pady=10)


button5 = Button(tab1, text="Reset", width=12, command=clear_entry_text, bg="#b9f6ca")
button5.grid(row=5, column=1, padx=10, pady=10)

button6 = Button(tab1, text="Clear Result", width=12, command=clear_display_result)
button6.grid(row=5, column=2, padx=10, pady=10)

# Display Screen For Result
tab1_display = Text(tab1)
tab1_display.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Allows you to edit
tab1_display.config(state=NORMAL)

# FILE READING  AND PROCESSING TAB
l1 = Label(tab2, text="Open File To Process")
l1.grid(row=1, column=1)

displayed_file = ScrolledText(tab2, height=7)  # Initial was Text(tab2)
displayed_file.grid(row=2, column=0, columnspan=3, padx=5, pady=3)

# BUTTONS FOR SECOND TAB/FILE READING TAB
b0 = Button(tab2, text="Open File", width=12, command=openfiles, bg='#c5cae9')
b0.grid(row=3, column=0, padx=10, pady=10)

b1 = Button(tab2, text="Reset ", width=12, command=clear_text_file, bg="#b9f6ca")
b1.grid(row=3, column=1, padx=10, pady=10)

#b1a = Button(tab2, text="NLpiffy", width=12, command=nlpiffy_file, bg='blue', fg='#fff')
#b1a.grid(row=3, column=2, padx=10, pady=10)

b2 = Button(tab2, text="SUMMARIZE", width=12, command=get_file_tokens, bg='#03A9F4', fg='#fff')
b2.grid(row=4, column=0, padx=10, pady=10)

b3 = Button(tab2, text="PREDICT", width=12, command=get_file_pos_tags, bg='#BB86FC')
b3.grid(row=4, column=1, padx=10, pady=10)



b6 = Button(tab2, text="Clear Result", width=12, command=clear_result)
b6.grid(row=5, column=1, padx=10, pady=10)

b7 = Button(tab2, text="Close", width=12, command=window.destroy)
b7.grid(row=5, column=2, padx=10, pady=10)

# Display Screen

# tab2_display_text = Text(tab2)
tab2_display_text = ScrolledText(tab2, height=10)
tab2_display_text.grid(row=7, column=0, columnspan=3, padx=5, pady=5)

# Allows you to edit
tab2_display_text.config(state=NORMAL)

window.mainloop()

