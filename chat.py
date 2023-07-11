import random
import json
import pandas as pd
import re
import torch
import joblib

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('file.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Define the data for the table
data_tab = {
    'tag': ['bedrooms', 'bathrooms', 'totallivingarea', 'totallotarea', 'floors', 'waterfront',
             'view', 'condition', 'grade', 'above', 'basement', 'built', 'renovated', 'zipcode',
             'latitude', 'longitude', 'livingarea', 'lotarea'],
    'value': [''] * 18   # Empty values for the 'value' column
}
tagss = data_tab['tag']
# Create the DataFrame
df = pd.DataFrame(data_tab)

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "HouseBot"

def get_response(msg):
    print(df)
    if "restart" in msg:
        df['value'] = [''] * len(df)
        return "Restarting... Please give me the tag and the value of the feature you wish to start with!"
    # Split the sentence into words
    elif  any(re.findall(r'\d+', msg)):
        if any(tag.lower() in msg.lower() for tag in tagss):
            words = msg.split()

            # Iterate over the words and check for tags
            for i in range(len(words)):
                # Check if the word is a tag
                tag = words[i].strip().lower()
                if tag in df['tag'].str.lower().values:
                    # Check the entire sentence for a corresponding value
                    value = None
                    for j in range(len(words)):
                        word = words[j].strip()
                        # Skip the current word if it is the tag
                        if j == i:
                            continue
                        # Check if the word is a valid value
                        if re.match(r'-?\d+(?:[-.]?\d+)*$', word):
                            if tag == "waterfront" :
                                if float(word) in [0, 1] :
                                    value = float(word)
                                    break
                                else :
                                    return "Give a waterfront value 0 for no/ 1 for yes"
                            elif tag == "view" :
                                if float(word) <=5 and float(word) >= 0  :
                                    value = float(word)
                                    break
                                else :
                                    return "Give a value between 0 and 5 for the view!"
                            elif tag == "condition" :
                                if float(word) <=5 and float(word) > 0  :
                                    value = float(word)
                                    break
                                else :
                                    return "Give a value between 1 and 5 for the house condition!"
                            elif tag == "grade" :
                                if float(word) < 14 and float(word) > 0  :
                                    value = float(word)
                                    break
                                else :
                                    return "Give a value between 1 and 13 for the grade!"
                            elif tag == "built" :
                                if float(word) < 2022 and float(word) >= 1900  :
                                    value = float(word)
                                    break
                                else :
                                    return "Give me a correct year of built (1900-2021)!"
                            elif tag == "renovated" :
                                if float(word) < 2022 :
                                    value = float(word)
                                    break
                                else :
                                    return "Give a correct year of renovation (< 2022)!"   
                            elif tag == "zipcode" :
                                if float(word) > 9999 and float(word) < 100000:
                                    value = float(word)
                                    break
                                else :
                                    return "The zipcode must contain 5 numbers!"
                                
                            else :
                                value = float(word)
                                break
                    if value is not None:
                        # Update the value in the table
                        df.loc[df['tag'].str.lower() == tag, 'value'] = value
                        print(f"Value for tag '{tag}' added to the table.")
                        break
                    else:
                        print(f"No value found for tag '{tag}'.")
                        if i > 0:
                            previous_tag = words[i-1].strip().lower()
                            print(f"The previous tag is '{previous_tag}'.")
                        break

            # Check for missing values in the table
            missing_tags = df[df['value'] == '']
            if len(missing_tags) > 0:
                missing_tag = missing_tags.iloc[0]['tag']
                print(f"The first missing value is for the tag: '{missing_tag}'.")
                return f"What about '{missing_tag}'?"
            else:
                print("No missing values found in the table.")
                # Load the saved model from file
                rf_model = joblib.load('random_forest_model.pkl')

                # Prepare the input data for prediction
                X_test = df['value'].astype(float).values.reshape(1, -1)

                # Make predictions on the input data
                predicted_price = rf_model.predict(X_test)

                # Print the predicted price
                print("Predicted Price:", predicted_price[0])
                return f"The predicted price of your house is: {predicted_price[0]}. If you with to restart again type \"restart\"!"
        else:
            i=0
            for index, row in df.iterrows():
                i=i+1
                tagd = row['tag']
                value = row['value']
                if value == '':
                    miss = tagd
                    break
            if tagd == "waterfront" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) in [0, 1] :        
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else:
                    return "Give a waterfront value 0 for no/1 for yes"
            elif tagd == "view" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) <= 5 and float(di[0]) >= 0 :
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "Give a value between 0 and 5 for the view!"
            elif tagd == "condition" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) <= 5 and float(di[0]) > 0 :
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "Give a value between 1 and 5 for the house condition!"
            elif tagd == "grade" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) < 14 and float(di[0]) > 0 :
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "Give a value between 1 and 13 for the grade!"
            elif tagd == "built" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) < 2022 and float(di[0]) > 1900 :
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "Give me a correct year of built (1900-2021)!"
            elif tagd == "renovated" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) < 2022 :
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "Give a correct year of renovation (< 2022)!"
            elif tagd == "zipcode" :
                di = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                if float(di[0]) > 9999 and float(di[0]) < 100000:
                    digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                    df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
                else :
                    return "The zipcode must contain 5 numbers!"
                
            else :
                digitss = re.findall(r'-?\d+(?:[-.]\d+)*', msg)
                df.loc[df['tag'].str.lower() == tagd, 'value'] = digitss
            #df.loc[df['tag'].str.lower() == miss, 'value'] = digitss
            if i < 18:
                while df.loc[i, 'value']!= '':
                    i = i+1
                    if i == 18:
                        break
                if i< 18:
                    miss_nextt = df.loc[i, 'tag']
                    return f"What about '{miss_nextt}'?"
                else:
                    # Load the saved model from file
                    rf_model = joblib.load('random_forest_model.pkl')

                    # Prepare the input data for prediction
                    X_test = df['value'].astype(float).values.reshape(1, -1)

                    # Make predictions on the input data
                    predicted_price = rf_model.predict(X_test)

                    # Print the predicted price
                    print("Predicted Price:", predicted_price[0])
                    return f"The predicted price of your house is: {predicted_price[0]}. If you with to restart again type \"restart\"!"
            else:
                # Load the saved model from file
                rf_model = joblib.load('random_forest_model.pkl')

                # Prepare the input data for prediction
                X_test = df['value'].astype(float).values.reshape(1, -1)

                # Make predictions on the input data
                predicted_price = rf_model.predict(X_test)

                # Print the predicted price
                print("Predicted Price:", predicted_price[0])
                return f"The predicted price of your house is: {predicted_price[0]}. If you with to restart again type \"restart\"!"

    else:
        # Process the message using the chatbot model
        sentence = tokenize(msg)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)

        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return random.choice(intent['responses'])

        return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
