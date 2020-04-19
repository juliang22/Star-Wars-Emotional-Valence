# Julian Grunauer —— 4/18/20 —— Computational Linguistics, Professor Solano —— Dartmouth College 
# This program uses the NRC-Emotion-Lexicon to estimate the levels of Darth Vader's emotions in 
# Episodes 4, 5, and 6 os Star Wars. Darth Vader's lines are collected, stemmed, referenced 
# against the NRC-Emotion-Lexicon, and the emotional values are then summed.
#
# SOURCES:
# https://stackoverflow.com/questions/40849273/python-3-match-names-from-a-text-file-and-print-them-out-line-by-line
# http://jonathansoma.com/lede/algorithms-2017/classes/more-text-analysis/nrc-emotional-lexicon/
# https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
# 

import re
import pandas as pd
import string
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

ps = PorterStemmer() 
i = 0
MOVIE_NUM = 4
PATTERN = r'(VADER	.*)'
MOV4 = { "anger": 0, "anticipation": 0, "disgust": 0, "fear": 0, "joy": 0, "negative": 0, "positive": 0, "sadness": 0, "surprise": 0, "trust": 0}
MOV5 = { "anger": 0, "anticipation": 0, "disgust": 0, "fear": 0, "joy": 0, "negative": 0, "positive": 0, "sadness": 0, "surprise": 0, "trust": 0}
MOV6 = { "anger": 0, "anticipation": 0, "disgust": 0, "fear": 0, "joy": 0, "negative": 0, "positive": 0, "sadness": 0, "surprise": 0, "trust": 0}
KEY = {0: "anger", 1: "anticipation", 2: "disgust", 3: "fear", 4: "joy", 5: "negative", 6: "positive", 7: "sadness", 8: "surprise", 9: "trust"}

# Parses through the NRC-Emotion-Lexicon, steans each word, and creates a dictionary of dictorionaries {word, {emotion: value, emotion: value....}}
NRC_DICT = {}
with open("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", 'r') as file:
    for line in file:
        # split each line into a list of [word, emotion, value]
        splits = line.split("\t", 3)
        # normalize each index
        line_arr = [splits[0], splits[1:2], splits[-1]]
        line_arr[1] = str(line_arr[1]).rstrip("]'")[2:]
        line_arr[2] = str(line_arr[-1]).rstrip("\n")
        if (line_arr[2] == '0'):
            line_arr[2] = 0
        else:
            line_arr[2] = 1
        # Create a dictiory for each word containing dictionaries for each emotion:value pair
        temp_word = ps.stem(line_arr[0])
        if temp_word in NRC_DICT:
            NRC_DICT[temp_word][line_arr[1]] = line_arr[2]
        else:
            inner_dict = {line_arr[1]: line_arr[2]}
            NRC_DICT[temp_word] = inner_dict

# Helper function to sum emotional content of each movie
def emo_counter(MOVIE_NUM, word):
    for c in range(9):
        if(NRC_DICT[word][KEY[c]] != 0):
            if (MOVIE_NUM == 4):
                MOV4[KEY[c]] += NRC_DICT[word][KEY[c]]
            if (MOVIE_NUM == 5):
                MOV5[KEY[c]] += NRC_DICT[word][KEY[c]]
            if (MOVIE_NUM == 6):
                MOV6[KEY[c]] += NRC_DICT[word][KEY[c]]

# Parses through Star Wars Episodes 4-6 scripts, tokenizes/stems each line of Darth Vader, and counts their emotional valence using
# the NRC-Emotion_Lexicon
while(i < 3):
    MOVIE = "sw" + str(MOVIE_NUM) + ".txt"
    with open(MOVIE, 'r') as file:
        for line in file:
            for match in re.finditer(PATTERN, line):
                no_punct = match.group().translate(str.maketrans('', '', string.punctuation))
                split = no_punct.split()
                for each_word in split:
                    each_word = ps.stem(each_word)
                    if(each_word in NRC_DICT):
                        emo_counter(MOVIE_NUM, each_word)
    MOVIE_NUM += 1
    i += 1

# Prints out values of Vader's emotion for each script and creates bar graphs for each movie
print("MOVIE 4: \n" + str(MOV4))
print("MOVIE 5: \n" + str(MOV5))
print("MOVIE 6: \n" + str(MOV6))

# data to plot
n_groups = 10
emo_4 = (MOV4["anger"], MOV4["anticipation"], MOV4["disgust"], MOV4["fear"], MOV4["joy"], MOV4["negative"], MOV4["positive"], MOV4["sadness"], MOV4["surprise"], MOV4["trust"])
emo_5 = (MOV5["anger"], MOV5["anticipation"], MOV5["disgust"], MOV5["fear"], MOV5["joy"], MOV5["negative"], MOV5["positive"], MOV5["sadness"], MOV5["surprise"], MOV5["trust"])
emo_6 = MOV6["anger"], MOV6["anticipation"], MOV6["disgust"], MOV6["fear"], MOV6["joy"], MOV6["negative"], MOV6["positive"], MOV6["sadness"], MOV6["surprise"], MOV6["trust"]

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.3
opacity = 0.8

rects1 = plt.bar(index, emo_4, bar_width,
alpha=opacity,
color='b',
label='Episode 4',
align='center')

rects2 = plt.bar(index + bar_width, emo_5, bar_width,
alpha=opacity,
color='g',
label='Episode 5',
align='center')

rects3 = plt.bar(index + bar_width + bar_width, emo_6, bar_width,
alpha=opacity,
color='r',
label='Episode 6',
align='center')

plt.xlabel('Emotion')
plt.ylabel('Emotional Valence')
plt.title('Darth Vader\'s Emotional Valence')
plt.xticks(index + bar_width, ('Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Negative', 'Positive', 'Sadness', 'Surprise', 'Trust'))
plt.legend()

plt.tight_layout()
plt.show()


