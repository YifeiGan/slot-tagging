
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sys

train_data_path = sys.argv[1]
test_data_path = sys.argv[2]
output_path = sys.argv[3]

df = pd.read_csv(train_data_path)

df["IOB Slot tags"] = df["IOB Slot tags"].fillna('')
df["IOB Slot tags"] = df["IOB Slot tags"].apply(lambda i: i.split() if isinstance(i, str) else [])
df["utterances"] = df["utterances"].apply(lambda x: x.split())
x = df["utterances"]
y = df["IOB Slot tags"]

# Encode words and labels to integers
word_encoder = LabelEncoder()
label_encoder = LabelEncoder()

# Flatten utterances and labels for encoding, then reshape
flat_utterances = [word for utterance in x for word in utterance]
flat_labels = [label for label_list in y for label in label_list]

word_encoder.fit(flat_utterances)
label_encoder.fit(flat_labels)

# Encode the utterances and labels
encoded_utterances = [word_encoder.transform(utterance) for utterance in x]
encoded_labels = [label_encoder.transform(label_list) for label_list in y]

class NERTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=160, hidden_dim=157):
        super(NERTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.fc(lstm_out)
        return tag_scores

def train_model(model, train_data, train_labels, epochs=19, learning_rate=0.000332992446379797):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        
        for sentence, tags in zip(train_data, train_labels):
            sentence_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            tags_tensor = torch.tensor(tags, dtype=torch.long).unsqueeze(0)  # Add batch dimension

            optimizer.zero_grad()
            tag_scores = model(sentence_tensor)
            loss = criterion(tag_scores.view(-1, tag_scores.shape[-1]), tags_tensor.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy for the batch
            predicted_tags = tag_scores.argmax(dim=-1).squeeze(0)
            correct += (predicted_tags == tags_tensor.squeeze(0)).sum().item()
            total += tags_tensor.size(1)  # Number of words in the sentence

        avg_loss = total_loss / len(train_data)
        accuracy = correct / total
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    return loss_history, accuracy_history

# Initialize and train the model
model = NERTagger(vocab_size=len(word_encoder.classes_), tagset_size=len(label_encoder.classes_))
loss_history, accuracy_history = train_model(model, encoded_utterances, encoded_labels)

#Generate predictions
df_test = pd.read_csv(test_data_path)
x_test = df_test["utterances"]

x_test = x_test.apply(lambda x: x.split())

def predict_tags(model, utterance, word_encoder, label_encoder):
    # Encode the utterance, using a fallback for unknown words
    encoded_utterance = []
    for word in utterance:
        if word in word_encoder.classes_:
            encoded_utterance.append(word_encoder.transform([word])[0])
        else:
            # Use a default index for unknown words; here, we'll use 0 (often mapped to '<UNK>')
            encoded_utterance.append(0)  # Ensure 0 is reserved for unknown in your vocabulary setup

    sentence_tensor = torch.tensor(encoded_utterance, dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # Make predictions with the model
    model.eval()
    with torch.no_grad():
        tag_scores = model(sentence_tensor)
        predicted_tags = tag_scores.argmax(dim=-1).squeeze(0).tolist()  # Get the predicted tag indices

    # Decode the predicted indices to readable labels
    decoded_tags = label_encoder.inverse_transform(predicted_tags)
    return list(zip(utterance, decoded_tags))

# Create a list to collect each row of data
submission_data = []

# Populate the list with the utterances and their corresponding predictions
for utterance in x_test:
    predictions = predict_tags(model, utterance, word_encoder, label_encoder)
    submission_data.append({
        'utterance': ' '.join(utterance),
        'IOB Slot tags': ' '.join([tag for _, tag in predictions])
    })

# Convert the list of dictionaries to a DataFrame
df_submission = pd.DataFrame(submission_data, columns=['ID', 'utterance', 'IOB Slot tags'])

for i in range(len(df_submission)):
    df_submission.loc[i, 'ID'] = i + 1

df_submission.to_csv(output_path, index=False)
print("Output file saved to:", output_path)

