import torch
import torch.nn as nn
import torch.optim as optim

# 1. Prepare raw text data
text_data = "the playful kitten chased the mouse"
words = text_data.split()

# 2. Build the Vocabulary (Map words to unique ID numbers)
vocab = {word: i for i, word in enumerate(set(words))}
vocab_size = len(vocab)
print("Vocabulary Mapping:", vocab)

# Convert our words into a sequence of numeric token IDs
token_ids = [vocab[word] for word in words]

# 3. Create Training Pairs (X = input word ID, Y = the next target word ID)
# Example: Input "the" -> Target "playful"
X_train = torch.tensor(token_ids[:-1], dtype=torch.long).unsqueeze(1) # Inputs
Y_train = torch.tensor(token_ids[1:], dtype=torch.long)              # Targets

# 4. Define the Text Prediction Model
class SimpleTextModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleTextModel, self).__init__()
        # Step 2: Turns token ID numbers into close parameter maps (vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Step 3: A standard layer to map word vector relationships
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        # Output layer: Predicts a score for EVERY word in our vocabulary
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x).squeeze(1) # Convert IDs to word vectors
        hidden = self.relu(self.linear1(embedded))
        scores = self.linear2(hidden)
        return scores

# 5. Initialize setup
model = SimpleTextModel(vocab_size=vocab_size, embedding_dim=8, hidden_dim=16)

# CrossEntropyLoss is the industry standard grader for word selection
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 6. The Learning Loop
for epoch in range(150):
    optimizer.zero_grad()
    
    predictions = model(X_train)       # Forward pass: score all vocabulary words
    loss = criterion(predictions, Y_train) # Compute error
    
    loss.backward()                    # Calculus: find the error direction
    optimizer.step()                   # Update word coordinates & layer weights

# 7. Test the Model: Predict what comes after "kitten"
model.eval()
with torch.no_grad():
    test_word = "kitten"
    test_id = torch.tensor([[vocab[test_word]]], dtype=torch.long)
    
    # Get scores for all possible vocabulary words
    output_scores = model(test_id)
    
    # Pick the word ID with the highest mathematical score
    predicted_id = torch.argmax(output_scores, dim=1).item()
    
    # Translate the ID back into a word string
    inverse_vocab = {i: word for word, i in vocab.items()}
    predicted_word = inverse_vocab[predicted_id]
    
    print(f"\nInput Word: '{test_word}' -> Predicted Next Word: '{predicted_word}'")
