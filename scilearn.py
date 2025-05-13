from keras.src.metrics.accuracy_metrics import accuracy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
#predictive model for classification from the reviews data sets downloaded to train and test and predict new data sets based on them.
df  = pd.read_csv('C:/Users/jerin/Downloads/sentimentlabelledsentences/sentimentlabelled/amazon_cells_labelled.txt', names = ['review', 'sentiment'], sep = '\t')
#converted txt file into a panda dataframe to do further analysis on the data by using read_csv() method
#we specify two columns one for hold reviews and another corresponding sentiment score

#once data is loaded as dataframe , now its time to split the data between two parts: one to train the predictive model and one to test its accuracy
reviews = df['review'].values
sentiments = df['sentiment'].values #returns numpy arrays obtained  via values property
reviews_train , reviews_test ,sentiment_train , sentiment_test = train_test_split(reviews , sentiments , test_size = 0.2  ,random_state = 500)

#transforming text into numerical feature vectors to test and train model. using bag of words(Bow) model which transform text to numerical using word frequency of number of times each word occurs in text
# use CountVectorizer() function to create bow matrix for text data. this converts the text into numerical and performs tokenization separating words and punctuations.
#custom tokenizer can be set using spaCy (natural language processing) package . using the default option below:
vectorizer = CountVectorizer()
vectorizer.fit(reviews) #to build the vocabulary of tokens found in the dataset
X_train = vectorizer.transform(reviews_train)
X_test = vectorizer.transform(reviews_test)

#now that the training and test sets are in the form of numerical vectors , can train and test the model.
# here using the LogisticRegression() classifier model to predict the sentiment of a review. this is basic but popular algorithm for solving classification problems.
classifier = LogisticRegression()
classifier.fit(X_train , sentiment_train)
#using fit() method to train model based on training data


#now to evaluate the accuracy of the model making prediction on the new data with set of labeled data or test set.
#Evaluate the model using test set
accuracy = classifier.score(X_test , sentiment_test)
print("Accuracy of the model is : ", accuracy)

#Once we have trained and tested our model, its ready to analyze new unlabelled data
# try by feeding the model new sample reviews
new_reviews = [ 'Old version of python useless', 'very good effort', 'Clear and concise']
X_new = vectorizer.transform(new_reviews)
print(classifier.predict(X_new))
#the predicted class sentiment for the reviews will be given 0 or 1 ( 0 for positive and 1 for negative)