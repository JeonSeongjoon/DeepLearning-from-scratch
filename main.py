import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

from model import VanillaRNN
from optimizer import SGD, Adam
from train import train, evaluate


def main():
    # ========== hyper parameters =========
    epochs = 10
    batch_size = 32
    input_size = 28  # 각 timestep의 feature 수 
    hidden_size = 64  # RNN hidden state 크기
    output_size = 10  
    # =====================================


    # Data loading
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print("=========== Data shape ============")
    print(f"Train data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print("===================================")

    # data preprocessing for proper use
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train_oh = np.eye(10)[y_train]
    y_test_oh = np.eye(10)[y_test]

    
    # model
    model = VanillaRNN(input_size, hidden_size, output_size)
    optimizer = Adam(lr=0.001)


    # train process
    print("Training Start")
    for epoch in range(epochs):
        train_loss = train(model, x_train, y_train_oh, batch_size, optimizer)
        
        train_acc = evaluate(model, x_train[:10000], y_train_oh[:10000], batch_size)
        test_acc = evaluate(model, x_test, y_test_oh, batch_size)
        
        print("[Epoch: {}] Train Loss: {:.4f} | Train ACC: {:.4f} | Test ACC: {:.4f} ".format(epoch+1, train_loss, train_acc, test_acc))
    
    print("Training Complete")


if __name__=="__main__":
    main()















