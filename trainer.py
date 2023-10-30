import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from json import loads
from chatmodel import NeuralNetwork
from nltk_stuffs import bow, stem_word, tokenize_sentence, to_be_ignored

intents = loads(open('intents.json').read()) # loading the intents.json

all_words = [] # this list holds all the words which is obtained by tokenizing the queries
tags = [] # contains the tags
query_and_tag=[] # holds a list of tuples that has a query and its respective tag

for intent in intents['intents']: # the large intents list which is a list of dictionaries in the json file
    current_tag = intent['tag']
    tags.append(current_tag)  # gets the tag individual dictionaries

    for query in intent['queries']: # loops through the queries list present in each dictionaries
        all_words.extend(tokenize_sentence(query)) # extends the all_words with the tokenized query list
        query_and_tag.append((tokenize_sentence(query), current_tag)) # appends a tuple that contains tokenized query list and its respective tag

all_words = [stem_word(word) for word in all_words if word not in to_be_ignored] # stemming individual words if that is not in to _be_ignored
all_words = sorted(set(all_words)) # sorting and removing duplicates
tags = sorted(tags)

x_train = [] # the x_train data is what holds the input data, like the tokened quries converted into binary format using the bow
y_train = [] # the y_train data holds the label of the tag in which the a bow vector presents, the label is a numerical value, here the label's index value is used.

for query, tag in query_and_tag:
    x_train.append(bow(query, all_words))

    label = tags.index(tag)
    y_train.append(label)

# our model maps a bow vector in x_train and its respective tag in the y_train, likewise it maps the remaining bow vectors and its corresponding tag, and draw meaningful mathametical connections between the the query(which is ofcourse binarified by bow) and its tag. 
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatbotDataset(Dataset):
    # we create a custom dataset that would help pytorch access our training data
    def __init__(self):
        self.n_samples = len(x_train) # no. of training samples
        self.x_data = x_train # input or featured training data
        self.y_data = y_train # the tag data in which each featured data belongs

    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i] # retrieves a particular training data (bow vector and its numerical tag)  
    
    def __len__(self):
        return self.n_samples # specifies the no. of training samples we have for pytorch, it helps it to know the size of dataset

batch_size = 8 # the batch size decides the no. of data samples present in each individual batches
hidden_size = 8 # hidden size referes to the no. of neurons present in each hidden neural layer 
learning_rate = 0.001 # this decides the rate of increase of model paramters, which is controlled by the optimizer. Higher learning rate means increasing of model paramters by larger value, smaller means, increase of model paramters in smaller steps. 

epochs = 1000 # this specifies how many times the model should be trained from the dataset, so in here, the model gets trained with the dataset for 1000 times. A epoch specifies the end of one complete training of model with the dataset. 

input_size = len(all_words) # size of the input size and the output sizes, and the model draws the relationships between these
output_size = len(tags) 

dataset = ChatbotDataset() # creates an instance of our chatbotdataset

training_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # in this training loader, we specified the dataset, the batch size, and set shuffle to true, which will shuffle the  data in random order while training, and then the num_workers helps us to utilize the system cpu efficiently. This training_loader, can be visulised as arry of tuples, each is a tensor itself containing 8 bow-ed query and its respective tags, 8 since we specified through the batch size (see __getitem__ in chatbotdataset class)

gpu_support = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checking for cuda-supported gpu. Increases training speed and  efficincy if available.

model = NeuralNetwork(input_size, hidden_size, output_size) # creating an instance of our neurl network by passing the required arguments.
model = model.to(gpu_support) # adds gpu support if available

loss_function = nn.CrossEntropyLoss() # Crossentropy loss calculates the loss through the dissmilarity between the model's predicted labels to the actual labels

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate) # we used the adam optimizer that would update the model parameters through modifiying the learning rate, based on the calculated gradients.

for epoch in range(epochs): # for each epoch
    for queries, labels in training_loader: # queries holds 8 bow-ed query as specified in batch size as a pytorch tensor, and the labels holds the label of corresponding tag to the bow-ed query
        queries = queries.to(gpu_support) # moves the query and labels to gpu if available
        labels = labels.to(gpu_support, dtype=torch.int64)
        output = model(queries) # the data samples are given to the model as input and the predicted  values are obtained as output. the data given to the model here flows with respective to the forward method specified in the neuralnetwork class.

        loss = loss_function(output, labels) # the loss_function calulates the difference or dissmiliarity between the predicted labels and the actual labels.

        optimizer.zero_grad() # cleaning the previously calculated gradients

        loss.backward() # this calculates the gradients of the loss. The gradients decides how much the model paramters should be changed to reduce the loss. Gradients represent how much the loss would change with respect to small changes in each parameter. This gradient is used in the optimizer to increase the model paramters

        optimizer.step()  # optimizer finally increases the model paramters based on the calculated gradients.


print(f"final result = epoch {epoch+1}/{epochs} , loss={loss.item():.4f}")

# after training we have to save our model's data in a file

model_data = {
    "model_state": model.state_dict(), # holds the model's structure and model paramters
    "hidden_size": hidden_size, # holds the hidden size i.e no. of neurons in each hidden layer
    "input_size" : input_size, # length of input i.e all_words
    "output_size": output_size, # length of output i.e tags
    "all_words": all_words, 
    "tags" : tags
}

FILE_NAME = "chatbot_data.pth" # some chatbot name
torch.save(model_data, FILE_NAME) # saving the trained model with its data in a file