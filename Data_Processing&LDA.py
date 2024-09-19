import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pip install pyLDAvis==3.4.1

# Part 1 Data Cleaning
df=pd.read_csv('Herborist_JDComment_data.csv')
df.head()
df.info()
#19201 * 8 
print(df[df.isnull().T.any()])
#No NaNs in the dataset
df_2=df.drop_duplicates()
df_2=df_2.reset_index(drop=True)
df_2.info()
#Drop duplicates，the dataset drops from 19201 to 11298
df_2[df_2['评论内容']=='此用户未填写评价内容']
#308 comments with no content

#Drop invalid comments
df_2[df_2['评论内容']=="此用户未填写评价内容"] = ""
print(len(df_2[df_2['评论内容']!=""])) 
#Count valid comments
#valid comments: 10990
import re
    
def filter_emoji(desstr,restr=''):
    '''
     filter emoji
    '''
    desstr=str(desstr)
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)

df_2['无表情评论']=df_2['评论内容'].apply(filter_emoji)
df_2.head()
df_2.to_csv("Cleaned_JDComments.csv",index=False)
df_3=df_2.drop(['评论内容'],axis=1)
df_3.to_csv("Emojifree_Cleaned_JDComments.csv",index=False)

# Part 2 Data Visualization and Analysis
from numpy import array
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

#Deal with product type
data=df_2[df_2['护肤水型号']!='']
value=array(data['护肤水型号'])
#Integer coding
product_type_encode = LabelEncoder()
integer_type_encoded = product_type_encode.fit_transform(value)
print(integer_type_encoded)
product_type = product_type_encode.classes_
print(product_type_encode.classes_)

# Part3 Process purchase time

hourList = []
month = []
year = []
useLessData = 0
for i in df_2['购买时间']:
    try:
        #print(str(i))
        hourList.append(str(i).split(' ')[1].split(':')[0])
        month.append(str(i).split(' ')[0].split('-')[1])
        year.append(str(i).split(' ')[0].split('-')[0])
    except:
        useLessData += 1
print('无效数据有%s条'%useLessData)
product_year_encode = LabelEncoder()
year_encoded = product_year_encode.fit_transform(year)
print(product_year_encode.classes_)
print(year_encoded)

## Part 3.1 Analyze the buying period

import random
from pyecharts.charts import Bar,Line,Grid
from pyecharts import options as opts
import os

bar=Bar()
attr = ["{}时".format(i) for i in range(24)]
v1 = [hourList.count(str(_).rjust(2,'0')) for _ in range(24)]
#Create barplot
#bar = Bar("商品购买时段",title_color='#FF0000',background_color='#7EC0EE')
bar.add_xaxis(attr)
bar.add_yaxis("",v1,
              itemstyle_opts=opts.ItemStyleOpts(color="#3b5cdd")
             )
bar.set_global_opts(title_opts=opts.TitleOpts(title="商品购买时段"))
#bar.add("", attr, v1, is_label_show=True, is_datazoom_show=True)

#Create lineplot
line=Line()
line.add_xaxis(attr)
line.add_yaxis("",v1,
               itemstyle_opts=opts.ItemStyleOpts(color="#8b9194")
              )
line.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#line.set_series_opts(itemstyle_opts=opts.ItemStyleOpts(color="#1E90FF"))

#Use Grid to overlay bar charts and line charts
grid=Grid()
grid.add(bar, grid_opts=opts.GridOpts(pos_left="5%", pos_right="20%"))
grid.add(line, grid_opts=opts.GridOpts(pos_left="5%", pos_right="20%"))
    
    
# render the chart
grid.render("bar_with_line.html")


#bar.render()
#os.system("render.html")
os.system("bar_with_line.html")

#Reference :https://blog.csdn.net/qq_27484665/article/details/116461440

"""
Analyzing consumers' purchases of goods in a single day,
it can be found that most consumers have peak consumption periods
in the morning (9-10 PM) and the evening (20-23 PM).
"""

## Part 3.2 Comment word cloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator

Comment_data = df_2['评论内容'].tolist()
dataStr = ','.join(Comment_data)
#print(','.join(data))

import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
 
backgroup_Image = plt.imread('phone.jpg') #Shrouding chart

font='/Users/eleanor.z/Desktop/MSYH.TTC'
 
#f = open('人工智能.txt','r').read()  #Generate word cloud documentation
wordcloud = WordCloud(
        background_color = 'white', #Background color, based on the background Settings of the image, the default is black
        mask = backgroup_Image, #Shrouding chart
        #font_path = 'C:\Windows\Fonts\STZHONGS.TTF',
        #font_path ='/Users/eleanor.z/Desktop/STZHONGS.TTF',
        font_path = font,
        width = 1000,
        height = 1200,
        margin = 2).generate(dataStr) # generate: All text can be automatically segmented
#parameter width，height，margin corresponding width pixel, length pixel, margin
 
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
 
#Save picture: The default path saved for this code
wordcloud.to_file('phoneComment_1.jpg')

"""
As can be seen from the figure, there are some invalid data that are not processed,
for example, hellip is invalid data.
And the influence of punctuation makes some sentences with more comments appear,
which is obviously not quite in line with the expected word-graph effect.
The following is to remove some redundant data, re-segmentation and survival of word cloud

"""
dataStr = dataStr.replace('hellip','')
dataStr = dataStr.replace('。','')
dataStr = dataStr.replace('，','').replace('！','').replace(',','').replace('&','').replace('...','')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import jieba
import os

backgroup_Image = plt.imread('phone.jpg') #Shrouding chart
wordlist = jieba.cut(dataStr, cut_all=False)
word_string = " ".join(wordlist)
font='/Users/eleanor.z/Desktop/MSYH.TTC'
wordcloud = WordCloud(font_path=font, background_color="white",mask = backgroup_Image, width=1000, height=860, margin=2).generate(word_string)
import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
wordcloud.to_file('phoneComment_2.jpg')

"""
From the word cloud drawn after the review cleaning, it can be seen that the high-frequency words "hydration",
"very good", "effect", "moisturizing", "non-greasy", "easy to use", "texture", "fragrance", "suitable for (my) skin"
in the review all point to the efficacy of the product itself, reflecting consumers' attention to the value of the product;
The frequency of "Herborist" is relatively low, which shows that consumers attach more importance to the efficacy of the product itself than the brand.
"""

wordlist = jieba.cut(dataStr, cut_all=False)
counts={}
for word in wordlist:
    if len(word)==1:
        #Exclude the result of a single word segmentation
        continue
    else:
        counts[word]=counts.get(word,0)+1 #dict

hist=list(counts.items())#form a list
hist.sort(key=lambda x:x[1],reverse=True)
for i in range(20): #Output the top 20 high-frequency words of the text
    word,count=hist[i]
    print("{:<10}{:>5}".format(word,count))

"""
    The statistics of the top 20 high-frequency words in the review further verified the above conclusions.
    That is, the customer's value is reflected in the comments: product value > brand value > Relationship value
"""


# Part 4 Text mining - Building sentiment analysis models

"""
Technical logic:
1. Construction of data set The data set was obtained by crawler technology and divided into tables according to user name, content,
comment time, comment type and comment Angle. The dataset consists of 8361 reviews, 6361 favorable reviews, 1000 medium reviews and 1000 negative reviews.
Here are all the reviews of a certain phone on the e-commerce platform.
To accomplish the affective attitude analysis task, the comment section of the dataset and the comment type
(0: negative, 1: positive, 2: neutral) were used.
The actual task is text classification.
2. Build pre-trained word vector model Word2Vec
i. Import Word2Vec model template (no weight)
ii from third-party library gensim.
Use Pandas to read the.xlsx table's comment section
iii.JieBa segmentation to get the output of each sentence segmentation: a sequential word array.
iv. Remove Spaces from the word segmentation array and separate by newlines to get the sentence format before the input model.
v. Train the Word2Vec model and produce the word2vec.model


"""

df_3.head()

##Segmentation of text data in batches
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

df_3["评论分词"] = df_3['无表情评论'].apply(chinese_word_cut)

df_3.head()

"""
Secondly, based on the bag of words model (without considering the order, grammar and relevance of words),
text vectorization is realized by converting keywords into features and text into multidimensional vector representation.
We imported the software packages TfidfVectorizer and CountVectorizer, and then carried out keyword extraction and vector conversion after limiting the number of feature keywords:


"""

##Text vectorization: Features that convert keywords into numerical representations
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#Limit the number of features
n_features = 1000
#Keyword extraction and vector conversion
tf_vectorizer = CountVectorizer(strip_accents = 'unicode',
                                max_features=n_features,
                                stop_words='english',
                                max_df = 0.5,
                                min_df = 10)
tf = tf_vectorizer.fit_transform(df_3["评论分词"])

#Import LatentDirichletAllocation，limit the topic number，then extract topics through LDA：

#Import LDA topic model
from sklearn.decomposition import LatentDirichletAllocation
#limit number of topics
n_topics = 5
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
lda.fit(tf)

#Next, we define a function that takes the model, the feature name, and the number of keywords as parameters, and displays the first few keywords in each topic.
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

#Then call the function to output the keyword table of each topic in turn:
tf_feature_names = tf_vectorizer.get_feature_names_out()
print_top_words(lda, tf_feature_names, n_top_words)

"""
After the keyword table output is correct, we try to visualize the extracted theme:
first import the theme model visualization pyLDAvis library and machine learning sklearn library,
and then call enable notebook to display the visualization results in the notebook.
"""

import pyLDAvis
import pyLDAvis.lda_model
pyLDAvis.enable_notebook()
data = pyLDAvis.lda_model.prepare(lda, tf, tf_vectorizer)
pyLDAvis.show(data)
