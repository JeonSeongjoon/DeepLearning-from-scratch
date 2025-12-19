import numpy as np

def train(model, x_data, y_data, batch_size, optimizer):
    total_loss = 0
    num_batches = 0
    
    indices = np.random.permutation(len(x_data))
    
    for i in range(0, len(x_data), batch_size):
        batch_indices = indices[i:i+batch_size]
        x_batch = x_data[batch_indices]
        y_batch = y_data[batch_indices]
        
        # Forward
        loss = model.forward(x_batch, y_batch)
        
        # Backward
        model.backward()
        
        # Update
        optimizer.update(model.params, model.grads)
        
        # I reset RNN state to train the model newly for next batch 
        model.reset_state()
        
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, x_data, y_data, batch_size=100):
    correct = 0
    total = 0
    
    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i+batch_size]
        y_batch = y_data[i:i+batch_size]
        
        h_last = model.rnn_layer.forward(x_batch)
        output = model.NN.Affine.forward(h_last)
        
        preds = np.argmax(output, axis=1)
        labels = np.argmax(y_batch, axis=1)
        
        correct += np.sum(preds == labels)
        total += len(y_batch)
        
        model.reset_state()
    
    acc = correct / total
    return acc