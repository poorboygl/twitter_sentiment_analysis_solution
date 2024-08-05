# #pip install googledrivedownloader
# from google_drive_downloader import GoogleDriveDownloader as gdd

# gdd.download_file_from_google_drive(file_id='1e1uIwcJ1-MviSn9yk_ldPGffDWVp6yK_',
#                                     dest_path='./twitter_sentiment_analysis_3cls_dataset.zip',
#                                     unzip=True)
# #pip install nltk
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import re
# import nltk
# nltk.download('stopwords')

# from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer

# dataset_path = 'Twitter_Data.csv'
# df = pd.read_csv(
#     dataset_path
# )
# # print(df)
# # print(df.info())
# # print(df.describe())

# null_rows = df.isnull().any(axis=1)

# df = df.dropna()

# def text_normalize(text):
#     # Lowercasing
#     text = text.lower()

#     # Retweet old acronym "RT" removal
#     text = re.sub(r'^rt[\s]+', '', text)

#     # Hyperlinks removal
#     text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

#     # Punctuation removal
#     text = re.sub(r'[^\w\s]', '', text)

#     # Remove stopwords
#     stop_words = set(stopwords.words('english'))
#     words = text.split()
#     words = [word for word in words if word not in stop_words]
#     text = ' '.join(words)

#     # Stemming
#     stemmer = SnowballStemmer('english')
#     words = text.split()
#     words = [stemmer.stem(word) for word in words]
#     text = ' '.join(words)

#     return text

# text = """We love this! Would you go?
# #talk #makememories #unplug
# #relax #iphone #smartphone #wifi #connect...
# http://fb.me/6N3LsUpCu
# """
# text = text_normalize(text)
# print(text)

# #Post-processed data
# df['clean_text'] = df['clean_text'].apply(lambda x: text_normalize(x))
# df.to_csv('output_file.csv', index=False)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
#import sys

#np.set_printoptions(threshold=sys.maxsize)
df = pd.read_csv('output_file.csv')
df = df.dropna()
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text']).toarray()
print(X.shape)
print(type(X))


#Export CSV
#df = pd.DataFrame(X)
#df.to_csv('X.csv',index=False)

#add bias = 1
intercept = np.ones((
    X.shape[0], 1)
)

X_b = np.concatenate(
    (intercept, X),
    axis=1
)

n_classes = df['category'].nunique()
n_samples = df['category'].size

y = df['category'].to_numpy() + 1
y = y.astype(np.uint8)
#y.shape = (162902,)
y_encoded = np.array(
    [np.zeros(n_classes) for value in range(n_samples)]
)
#y_encoded = (162902, 3)
y_encoded[np.arange(n_samples), y] = 1


#Create train, val, test set 
val_size = 0.3
test_size = 0.1
random_state = 2
is_shuffle = True

#X_b = (162902, 2001)
#X_train = (114031, 2001)
#X_val = (43983, 2001)
#y_train= (114031, 3)
#y_val =  (43983, 3)

X_train, X_val, y_train, y_val = train_test_split(
    X_b, y_encoded,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle
)

X_val, X_test, y_val, y_test = train_test_split(
    X_val, y_val,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

print(f'Number of training samples: {X_train.shape[0]}')
print(f'Number of val samples: {X_val.shape[0]}')
print(f'Number of test samples: {X_test.shape[0]}')

#1. softmax function
def softmax(z):
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=1)[:, None]

#3. Cross-entropy loss function
def compute_loss(y_hat, y):
    n = y.size
    return (-1 / n) * np.sum(y * np.log(y_hat))

#2. Hypothesis function
def predict(X, theta):
    z = np.dot(X, theta)
    y_hat = softmax(z)
    return y_hat

#4. Gradient function
def compute_gradient(X, y, y_hat):
    n = y.size
    return np.dot(X.T, (y_hat - y)) / n

#5. Update weights function
def update_theta(theta, gradient, lr):
    return theta - lr * gradient

#6. Accuracy function  
def compute_accuracy(X, y, theta):
    y_hat = predict(X, theta)
    acc = (np.argmax(y_hat, axis=1) == np.argmax(y, axis=1)).mean()
    return acc

## 8. Training
lr = 0.1
epochs = 200
batch_size = X_train.shape[0]
n_features = X_train.shape[1]

np.random.seed(random_state)
theta = np.random.uniform(
    size=(n_features, n_classes)
)

train_accs = []
train_losses = []
val_accs = []
val_losses = []

for epoch in range(epochs):
    train_batch_losses = []
    train_batch_accs = []
    val_batch_losses = []
    val_batch_accs = []

    for i in range(0, X_train.shape[0], batch_size):
        X_i = X_train[i:i+batch_size]
        y_i = y_train[i:i+batch_size]

        #y_hat = (114031, 3)
        y_hat = predict(X_i, theta)
        #train_loss = float
        train_loss = compute_loss(y_hat, y_i)
        gradient = compute_gradient(X_i, y_i, y_hat)
        #gradient = (2001,3)
        theta = update_theta(theta, gradient, lr)
        #theta = (2001,3)
        train_batch_losses.append(train_loss)

        train_acc = compute_accuracy(X_train, y_train, theta)
        train_batch_accs.append(train_acc)

        y_val_hat = predict(X_val, theta)
        val_loss = compute_loss(y_val_hat, y_val)
        val_batch_losses.append(val_loss)

        val_acc = compute_accuracy(X_val, y_val, theta)
        val_batch_accs.append(val_acc)

    train_batch_loss = sum(train_batch_losses) / len(train_batch_losses)
    val_batch_loss = sum(val_batch_losses) / len(val_batch_losses)
    train_batch_acc = sum(train_batch_accs) / len(train_batch_accs)
    val_batch_acc = sum(val_batch_accs) / len(val_batch_accs)

    train_losses.append(train_batch_loss)
    val_losses.append(val_batch_loss)
    train_accs.append(train_batch_acc)
    val_accs.append(val_batch_acc)

    print(f'\nEPOCH {epoch + 1}:\tTraining loss: {train_batch_loss:.3f}\tValidation loss: {val_batch_loss:.3f}')

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
ax[0, 0].plot(train_losses, color='green')
ax[0, 0].set(xlabel='Epoch', ylabel='Loss')
ax[0, 0].set_title('Training Loss')

ax[0, 1].plot(val_losses, color='orange')
ax[0, 1].set(xlabel='Epoch', ylabel='Loss')
ax[0, 1].set_title('Validation Loss')

ax[1, 0].plot(train_accs, color='green')
ax[1, 0].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 0].set_title('Training Accuracy')

ax[1, 1].plot(val_accs, color='orange')
ax[1, 1].set(xlabel='Epoch', ylabel='Accuracy')
ax[1, 1].set_title('Validation Accuracy')

plt.savefig("result.pdf")