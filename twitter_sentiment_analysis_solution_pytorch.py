
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# dataset_path = 'Twitter_Data.csv'
# df = pd.read_csv(
#     dataset_path
# )
# # print(df)
# # print(df.info())
# # print(df.describe())

# #null_rows = df.isnull().any(axis=1)

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

dataset_path = 'output_file.csv'
df = pd.read_csv(
    dataset_path
)

df = df.dropna()
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['clean_text']).toarray()

# intercept = np.ones((
#     X.shape[0], 1)
# )

# X_b = np.concatenate(
#     (intercept, X),
#     axis=1
# )

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

X = torch.tensor(X, dtype= torch.float32)
y_encoded = torch.tensor(y_encoded, dtype = torch.float32)
val_size = 0.3
test_size = 0.1
random_state = 2
is_shuffle = True

torch.manual_seed(random_state)
if torch.cuda.is_available():
    torch.cuda.manual_seed(random_state)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded,
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

input_dim = X_train.shape[1]
output_dim = y_train.shape[1]

model = nn.Linear(
    input_dim,
    output_dim,
    bias= True
)

def compute_accuracy(y_hat, y_true):
    _, y_hat = torch.max(y_hat, dim= 1)
    _, y_true = torch.max(y_true, dim = 1)
    correct = (y_hat == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters() ,lr=0.1)
num_epoch = 200

train_accs = []
train_losses = []
val_accs = []
val_losses = []

for i in range(num_epoch):
    model.train() 
    # Zero the gradients
    # Forward pass
    y_hat = model(X_train)
    # Compute loss
    train_loss = criterion(y_hat, y_train)
    train_losses.append(train_loss.item())

    train_acc = compute_accuracy(y_hat, y_train)
    train_accs.append(train_acc)
    optimizer.zero_grad()
    # Backward pass and optimization
    train_loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        y_val_hat  = model(X_val)
        val_loss = criterion(y_hat, y_train)
        val_losses.append(val_loss.item())

        val_acc = compute_accuracy(y_val_hat, y_val)
        val_accs.append(val_acc)

    print(f'\nEPOCH {i+ 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}')

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

plt.savefig("result_pytorch.pdf")       





