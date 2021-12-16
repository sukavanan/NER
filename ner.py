import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.utils import class_weight
import warnings
import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau as lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings('ignore')

#config variables
train_path = "data/train"
dev_path = "data/dev"
test_path = "data/test"
glove_path = "data/glove.6B.100d.gz"
unk = "<unk>"
pad = "<pad>"
num = "<num>"
sym = "<sym>"
max_len = 128
batch_size = 8
numbers = ['one','two','three','four','five', 'six','seven','eight','nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'zero', 'hundred', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion']

def is_number(s):
    try:
        if "," in s:
            s = s.replace(",", "")
        if ":" in s:
            s = s.replace(":", "")
        if "-" in s:
            s = s.replace("-", "")
        if "/" in s:
            s = s.replace("/", "")
        if "." in s:
            s = s.replace(".", "")
        if s.lower() in numbers:
            return True
        float(s)
        return True
    except ValueError:
        return False
def is_symbol(s):
    flag = 1
    for char in s:
        if char.isalnum():
            flag = 0
            break
    if flag == 1:
        return True
    else:
        return False

#Function to read the datafile in the "path" line by line and return a list of lists 
#after removing trailing white spaces and empty lists
print("Reading data...")
def read_data(path):
    with open(path, "r") as file_obj:
        lines = file_obj.readlines()
    line_list = [line.rstrip().split() for line in lines]
    line_list = [x for x in line_list if x != []]
    return line_list

train_set = read_data(train_path)
dev_set = read_data(dev_path)
test_set = read_data(test_path)

vocab = {}
tag_to_idx = {}
idx_to_tag = {}
just_tags = []
idx = 0
threshold = 1

#creating vocabulary from the training_data for task1,
#removed the process of deleting words that are rare to increase performance
#also, creating the tag to idx and idx to tag dictionaries
for line in train_set:
    word = line[1]
    tag = line[2]
    if is_number(word):
        word = num
    if word in vocab:
        vocab[word] += 1
    else:
        vocab[word] = 1
    if tag not in tag_to_idx:
        tag_to_idx[tag] = idx
        idx_to_tag[idx] = tag
        idx += 1
    just_tags.append(tag_to_idx[tag])
# for word, freq in  list(vocab.items()):
#     if freq <= threshold:
#         del vocab[word]
vocab["<unk>"] = 1

#creating word to idx and idx to word dictionaries to represent words as indices
word_to_idx = {}
idx_to_word = {}
idx = 1
for word in vocab:
    if word not in word_to_idx:
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        idx += 1
word_to_idx[pad] = 0
idx_to_word[0] = pad
pad_idx = 0

#function to return a numpy array of indices of the words padded to max_len = 128
def dataset_creation(sentence, test = False):
    global word_to_idx, tag_to_idx, unk, max_len
    sen_len = len(sentence)
    pad_len = max_len - sen_len
    words = []
    tags = []
    if test:
        for idx, word in sentence:
            if word not in word_to_idx:
                if is_number(word):
                    word = num
                else:
                    word = unk
            words.append(word_to_idx[word])
        words = np.array(words)
        pad_seq = pad_idx * np.ones(pad_len)
        words = np.concatenate((words, pad_seq), axis = 0)
        return words
    else:
        for idx, word, tag in sentence:
            if word not in word_to_idx:
                if is_number(word):
                    word = num
    #             elif word.isupper():
    #                 temp = word[0] + word[1:].lower()
    #                 if temp in word_to_idx:
    #                     word = temp
    #                 else:
    #                     word = unk
                else:
                    word = unk
            words.append(word_to_idx[word])
            tags.append(tag_to_idx[tag])
        words = np.array(words)
        tags = np.array(tags)
        pad_seq = pad_idx * np.ones(pad_len)
        pad_tag = -1 * np.ones(pad_len)
        words = np.concatenate((words, pad_seq), axis = 0)
        tags = np.concatenate((tags, pad_tag), axis = 0)
        return words, tags

#function to create the input and output data that is used to train the network
#function calls dataset_creation on each sentence
def create_lstm_ip(dataset, test = False):
    x_lstm = []
    y_lstm = []
    sentence = []
    for i in range(len(dataset)):
        if i == len(dataset) - 1 or (dataset[i][0] == '1' and i != 0):
            if test:
                x = dataset_creation(sentence, test)
                x_lstm.append(x)
            else:
                x, y = dataset_creation(sentence, test)
                x_lstm.append(x)
                y_lstm.append(y)
            sentence = []
            sentence.append(dataset[i])
        else:
            sentence.append(dataset[i])
    if sentence != []:
        if test:
            last_x = dataset_creation(sentence, test)
            x_lstm.append(last_x)
        else:
            last_x, last_y = dataset_creation(sentence)
            x_lstm.append(last_x)
            y_lstm.append(last_y)
    if test:
        return np.array(x_lstm)
    else:
        return np.array(x_lstm), np.array(y_lstm)

print("Creating dataset for task 1...")
x_lstm_train, y_lstm_train = create_lstm_ip(train_set, False)
x_lstm_dev, y_lstm_dev = create_lstm_ip(dev_set, False)
x_lstm_test = create_lstm_ip(test_set, True)

#convertin the arrays into tensors
x_lstm_train, x_lstm_dev, x_lstm_test = torch.LongTensor(x_lstm_train), torch.LongTensor(x_lstm_dev), torch.LongTensor(x_lstm_test)
y_lstm_train, y_lstm_dev = torch.LongTensor(y_lstm_train), torch.LongTensor(y_lstm_dev)

#function to return the original lengths of the padded sentences in a batch
#function returns a tensor of batch_size containing the lengths of the corresponding sentence
def get_lengths(seq, idx):
    lens = []
    for x in seq:
        length = 0
        for i in range(len(x)):
            if x[i] == idx:
                break
            length += 1
        lens.append(length)
    return torch.Tensor(lens)

#dataset class definition
class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

class data(Dataset):
    def __init__(self, inputs, transform = None):
        self.data = inputs
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        inputs = self.data[index][0]
        label = self.data[index][1]
        if self.transform is not None:
            inputs = self.transform(inputs)
            
        return inputs, label

#creating the data loaders
lstm_train_dataset = TensorDataset(x_lstm_train, y_lstm_train)
lstm_train_dataset = data(lstm_train_dataset)
lstm_dev_dataset = TensorDataset(x_lstm_dev, y_lstm_dev)
lstm_dev_dataset = data(lstm_dev_dataset)

lstm_train_loader = DataLoader(lstm_train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)
lstm_dev_loader = DataLoader(lstm_dev_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

#initializing network variables
input_dim = len(word_to_idx)
embed_dim = 100
hidden_dim = 256
linear_dim = 128
output_dim = len(tag_to_idx)
pad_idx = word_to_idx[pad]
class_weights = class_weight.compute_class_weight('balanced', np.unique(just_tags), just_tags)

#accuracy function to determine sentence level accuracy in a batch
def accuracy(pred, targ):
    pred = pred.argmax(dim = 1, keepdim = True)
    non_pad_elements = (targ != -1).nonzero()
    correct = pred[non_pad_elements].squeeze(1).eq(targ[non_pad_elements])
    return correct.sum() / torch.FloatTensor([targ[non_pad_elements].shape[0]]).to(device)

#bi-LSTM model with an embedding layer
class bLSTM(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, linear_dim, output_dim, pad_idx):
        super(bLSTM, self).__init__()
        self.embedding_dim = embed_dim
        self.embedding = torch.nn.Embedding(num_embeddings = input_dim, embedding_dim = embed_dim)
        self.blstm = torch.nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.33)
        self.linear = torch.nn.Linear(hidden_dim*2, linear_dim)
        self.elu = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_dim, output_dim)
    
    def forward(self, x):
        emb = self.embedding(x)
        lens = get_lengths(x, 0)
        packed = pack_padded_sequence(emb, lens, batch_first = True, enforce_sorted = False)
        blstm_out, _ = self.blstm(packed)
        blstm_out, _ = pad_packed_sequence(blstm_out, batch_first = True, padding_value = 0, total_length = 128)
        lin_out = self.elu(self.linear(blstm_out))
        class_out = self.classifier(lin_out)
        return class_out
    
    def init_weights(self):
        for name, param in self.named_parameters():
            torch.nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings(self, padding_idx):
        self.embedding.weight.data[padding_idx] = torch.zeros(self.embedding_dim)

#function to create the output file in the required format
def create_op_file(x, y, model, dataset, file, pad_idx, test = False):
    model.eval()
    line_num = 0
    global idx_to_word, idx_to_tag
    with torch.no_grad():
        with open(file, "w") as fp:
            for i in range(len(x)):
                idx = 1
                ip = x[i].to(device)
                ip = torch.unsqueeze(ip, 0)
                op = model(ip)
                op = op.view(-1, op.shape[-1])
                _, pred = torch.max(op, 1)
                if test:
                    for j in range(len(pred)):
                        if x[i][j] == pad_idx:
                            if i != len(x) - 1:
                                fp.write("\n")
                            break
                        pred_tag = int(pred[j].item())
                        z = dataset[line_num][1]
                        fp.write("{} {} {}\n".format(idx, z, idx_to_tag[pred_tag]))
                        line_num += 1
                        idx += 1
                else:
                    target = y[i]
                    for j in range(len(target)):
                        if target[j] == -1:
                            if i != len(x) - 1:
                                fp.write("\n")
                            break
                        pred_tag = int(pred[j].item())
                        targ_tag = int(target[j].item())
                        z = dataset[line_num][1]
                        fp.write("{} {} {}\n".format(idx, z, idx_to_tag[pred_tag]))
                        line_num += 1
                        idx += 1

print("Loading model for task 1...")
#loading the model to predict on given dataset and store it as a file
model = bLSTM(input_dim, embed_dim, hidden_dim, linear_dim, output_dim, pad_idx).to(device)
print(model)
model.load_state_dict(torch.load('model/blstm1.pt'))
print("Predicting and creating output file for dev set...")
create_op_file(x_lstm_dev, y_lstm_dev, model, dev_set, "outputs/dev1.out", pad_idx, False)
print("Done...")

del model

#loading the model to predict on given dataset and store it as a file
model = bLSTM(input_dim, embed_dim, hidden_dim, linear_dim, output_dim, pad_idx).to(device)
model.load_state_dict(torch.load('model/blstm1.pt'))
print("Predicting and creating output file for test set...")
create_op_file(x_lstm_test, None, model, test_set, "outputs/test1.out", pad_idx, True)
print("Done...")

del model

print("Moving on to task 2...")
#function to create the list of lists which just contain the words, from the given dataset
def create_corpus(dataset, test = False):
    sentences = []
    sent = []
    if test:
        for idx, word in dataset:
            if idx == '1' and sent != []:
                sentences.append(sent)
                sent = [word]
            else:
                sent.append(word)
        if sent != []:
            sentences.append(sent)
    else:
        for idx, word, tag in dataset:
            if idx == '1' and sent != []:
                sentences.append(sent)
                sent = [word]
            else:
                sent.append(word)
        if sent != []:
            sentences.append(sent)
    return sentences

print("Creating corpus for task 2...")
#creating a corpus that contains sentences from train, dev and test dataset
train_corpus = create_corpus(train_set, False)
dev_corpus = create_corpus(dev_set, False)
test_corpus = create_corpus(test_set, True)
corpus = train_corpus + dev_corpus + test_corpus

print("Loading the GloVe model...")
#loading the glove model
glove_w2v_file = 'data/glove.6B.100d.txt.word2vec'
glove2word2vec(glove_path, glove_w2v_file)
glove_vec = KeyedVectors.load_word2vec_format(glove_w2v_file)

#function to create a global vocabulary consisting of words from train, dev and test sets
#function creates the global word to idx and idx to word dictionaries
def create_global_word2idx(corpus):
    idx = 0
    global all_word_to_idx, all_idx_to_word
    for sent in corpus:
        for word in sent:
            if is_number(word):
                word = num
            if word not in all_word_to_idx:
                all_word_to_idx[word] = idx
                all_idx_to_word[idx] = word
                idx += 1
    all_word_to_idx[pad] = idx
    all_idx_to_word[idx] = pad

#function to create the weight matrix, for the embedding layer, from the glove vectors of each word in the corpus
#words that do not have a glove embedding are assigned random embedding from a normal distribution
#words are first converted into lower case and then assigned an embedding which is stored in a embedding dictionary
#two words, capitalised and lower case are assigned the same embedding initially but they differ by their indices
#since the embedding layer with the weight matrix is trainable, at the end of the training,
#the embeddings of both those words should look different if they are not semantically similar to each other
def create_weight_matrix(weight_matrix, model):
    global all_word_to_idx, embedding_dict
    for word, idx in all_word_to_idx.items():
#         embed = np.zeros(100, dtype = float)
        word = word.lower()
        if word in embedding_dict:
            weight_matrix[idx] = embedding_dict[word]
        else:
            try:
                weight_matrix[idx] = model[word]
            except KeyError:
                rand_embed = np.random.normal(scale = 0.6, size = (100,))
                weight_matrix[idx] = rand_embed
                embedding_dict[word] = rand_embed
    
    return weight_matrix

#function to pad the sentences
def pad_glove(sentence):
    global max_len, glove_pad
    diff = max_len - len(sentence)
    sentence = np.concatenate((sentence, glove_pad * np.ones(diff, dtype = float)))
    return sentence

#function to create the dataset for bi-LSTM with GloVe embeddings
def create_glove_data(sentences, test = False):
    global all_word_to_idx
    glove_sentences = []
    glove_sent = []
    if test:
        for idx, word, in sentences:
            if is_number(word):
                word = num
            if idx == '1' and glove_sent != []:
                temp = np.array(glove_sent)
                temp = pad_glove(temp)
                glove_sentences.append(temp)
                glove_sent = [all_word_to_idx[word]]
            else:
                glove_sent.append(all_word_to_idx[word])
        if glove_sent != []:
            temp = np.array(glove_sent)
            temp = pad_glove(temp)
            glove_sentences.append(temp)
    else:
        for idx, word, tag in sentences:
            if is_number(word):
                word = num
            if idx == '1' and glove_sent != []:
                temp = np.array(glove_sent)
                temp = pad_glove(temp)
                glove_sentences.append(temp)
                glove_sent = [all_word_to_idx[word]]
            else:
                glove_sent.append(all_word_to_idx[word])
        if glove_sent != []:
            temp = np.array(glove_sent)
            temp = pad_glove(temp)
            glove_sentences.append(temp)
    return np.array(glove_sentences)

#creating the global corpus and dictionaries
all_word_to_idx = {}
all_idx_to_word = {}
create_global_word2idx(corpus)
glove_pad = all_word_to_idx[pad]

print("Creating weight matrix...")
#creating the weight matrix
weight_matrix = np.zeros((len(all_word_to_idx), 100), dtype = float)
embedding_dict = {}
embedding_dict[pad] = np.zeros(100, dtype = float)
weight_matrix = create_weight_matrix(weight_matrix, glove_vec)

train_sent = create_glove_data(train_set, False)
dev_sent = create_glove_data(dev_set, False)
test_sent = create_glove_data(test_set, True)

glove_train_x, glove_dev_x, glove_test_x = torch.LongTensor(train_sent).to(device), torch.LongTensor(dev_sent).to(device), torch.LongTensor(test_sent).to(device)

#since the tags and their indices are the same in both tasks, the tag tensors can be reused
lstm_glove_train_dataset = TensorDataset(glove_train_x, y_lstm_train)
lstm_glove_train_dataset = data(lstm_glove_train_dataset)
lstm_glove_dev_dataset = TensorDataset(glove_dev_x, y_lstm_dev)
lstm_glove_dev_dataset = data(lstm_glove_dev_dataset)

lstm_glove_train_loader = DataLoader(lstm_glove_train_dataset, batch_size = batch_size, drop_last = True, shuffle = True)
lstm_glove_dev_loader = DataLoader(lstm_glove_dev_dataset, batch_size = batch_size, drop_last = True, shuffle = True)

input_dim = len(all_word_to_idx)
embed_dim = 100
hidden_dim = 256
linear_dim = 128
output_dim = len(tag_to_idx)
class_weights = class_weight.compute_class_weight('balanced', np.unique(just_tags), just_tags)

#function to create the embedding layer and load the weights from the weight matrix
#returns an embedding layer
def create_embedding(input_dim, embed_dim, pad_idx, weight_matrix):
    weight_matrix = torch.FloatTensor(weight_matrix).to(device)
    embedding = torch.nn.Embedding(num_embeddings = input_dim, embedding_dim = embed_dim, padding_idx = pad_idx)
    embedding.load_state_dict({'weight': weight_matrix})
    return embedding

#bi-LSTM with GloVe embeddings model
#initialising weights of the network didn't improve performance
class glove_bLSTM(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, linear_dim, output_dim, pad_idx, weight_matrix):
        super(glove_bLSTM, self).__init__()
        self.pad_idx = pad_idx
        self.embedding = create_embedding(input_dim, embed_dim, pad_idx, weight_matrix)
        self.blstm = torch.nn.LSTM(input_size = embed_dim, hidden_size = hidden_dim, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.33)
        self.linear = torch.nn.Linear(hidden_dim * 2, linear_dim)
        self.elu = torch.nn.ELU()
        self.classifier = torch.nn.Linear(linear_dim, output_dim)
    
    def forward(self, x):
        emb = self.embedding(x)
        pad_idx = self.pad_idx
        lens = get_lengths(x, pad_idx)
        packed = pack_padded_sequence(emb, lens, batch_first = True, enforce_sorted = False)
        blstm_out, _ = self.blstm(packed)
        blstm_out, _ = pad_packed_sequence(blstm_out, batch_first = True, padding_value = 0, total_length = 128)
        lin_out = self.elu(self.linear(blstm_out))
        class_out = self.classifier(lin_out)
        return class_out

print("Loading model for task 2...")
model = glove_bLSTM(input_dim, embed_dim, hidden_dim, linear_dim, output_dim, glove_pad, weight_matrix).to(device)
print(model)
model.load_state_dict(torch.load('model/blstm2.pt'))
print("Predicting and creating output file for dev set...")
create_op_file(glove_dev_x, y_lstm_dev, model, dev_set, "outputs/dev2.out", glove_pad, False)
print("Done...")

del model

model = glove_bLSTM(input_dim, embed_dim, hidden_dim, linear_dim, output_dim, glove_pad, weight_matrix).to(device)
model.load_state_dict(torch.load('model/blstm2.pt'))
print("Predicting and creating output file for test set...")
create_op_file(glove_test_x, None, model, test_set, "outputs/test2.out", glove_pad, True)
print("Done...")