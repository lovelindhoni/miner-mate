from json import loads # to load the intents.json
import torch
import random # to randomnly choose a response based on query
from nltk_stuffs import tokenize_sentence, bow # for processing the user input
from chatmodel import NeuralNetwork

gpu_support = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # checking for cuda-supported gpu. Increases training speed and  efficincy if available.

intents = loads(open('intents.json').read()) # loading the intents.json

FILE_NAME = "chatbot_data.pth"
model_data = torch.load(FILE_NAME) # loading the saved model

model = NeuralNetwork(model_data["input_size"], model_data["hidden_size"], model_data["output_size"]) # creating an instance of our neural network with our trained model's data which we have saved

model = model.to(gpu_support) # adds gpu support if available

model.load_state_dict(model_data['model_state']) # loading the state of the saved trained model', i.e loading the trained model's paramters

model.eval() # model in evalutation mode, this specifies that we are evaluating our model after training

bot_name = "Prasath" 
print(f"Hi, I am {bot_name}. Lets's chat, type 'quit' to exit anytime")

while True:
    user_query = input("You: ")
    if user_query.lower() == "quit": # if user enters quit
        print("Thanks for chatting, I hope I was helpful to you")
        break; 

    user_query = tokenize_sentence(user_query) # tokenizing the user_query
    x = bow(user_query, model_data["all_words"]) # bow-ing the tokenized user-query
    x = x.reshape(1, x.shape[0]) # we are reshaping the 1d array to 2d array, since it is necessary for our model, because our model needs input as a array of batches, and x here denotes a array with a single batch.
    x = torch.from_numpy(x).to(gpu_support) # converting the numpy array to pytorch tensor
    output = model(x) # gives the tensor as input to our model
    predicted = (torch.max(output, dim=1))[1] # the torch max returns the max probability of tag and its label i.e its index in the tags list. This predicted variable stores the label of the tag with max probability in a special data type.

    predicted_tag = model_data["tags"][predicted.item()] # gets the predicted tag. The predicted.item() converts the special data type of predicted to integer value.
     
    probabilities = torch.softmax(output, dim=1) # gets the probability of all the tags in one dimension that is close to the corrrect tag
    prob = probabilities[0][predicted.item()] # this gets the probability of the of the predicted tag. predicted.item() converts the special type to integer

    if prob.item() < 0.75: # if the probability of the predicted tag is low
        print("Apologies, I do not understand..")
    else:
        for intent in intents['intents']: # looping through the json
            if predicted_tag == intent['tag']: 
                print(f"{bot_name}: {random.choice(intent['responses'])}") # the bot shows a random response to the user input based on the predicted tag.
                print("")

