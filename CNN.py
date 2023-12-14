import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import tensorflow as tf
import matplotlib.pyplot as plt


#preprocess data, expects the training data to be saved separately in a file with the name shown
dataframe = pd.read_csv("preprocessed_data.csv", header=0)

dataframe['Winner'] = dataframe['Winner'].apply(lambda x: 1 if x == 'Red' else 0).astype(int)
dataframe['title_bout'] = dataframe['title_bout'].apply(lambda x: 1 if x == True else 0).astype(int)

# Separate features and target variable
X = dataframe.drop('Winner', axis=1)
y = dataframe['Winner']

# Separate into training, validation, and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=.2, random_state=42)

#call back for tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",update_freq=1,histogram_freq=1)

'''
This model consists of a 1D convolutional layer followed by max pooling, 
a flattening layer, and two fully connected layers for binary classification. The 
ReLU activation function is used in the convolutional and dense layers, while 
the output layer uses a sigmoid activation to produce the final probability prediction.

'''
model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(layers.MaxPooling1D(2))
# Add more layers as needed
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification (winner/loser)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = 15

model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_validate, y_validate), callbacks=[tensorboard_callback])


#code for visualizing the convolutional filters post training
weights = model.layers[0].get_weights()[0]

if len(weights.shape) == 3:
    fig, axs = plt.subplots(1, weights.shape[2], figsize=(15, 5))  # Adjust figsize as needed

    for i in range(weights.shape[2]):
        axs[i].imshow(weights[:, :, i], cmap='gray', vmin=weights.min(), vmax=weights.max())
        axs[i].set_title(f'Filter {i+1}')
        axs[i].axis('off')

        for j in range(weights.shape[0]):
            for k in range(weights.shape[1]):
                axs[i].text(k, j, f'{weights[j, k, i]:.2f}', ha='center', va='center', fontsize=6, color='r')

    plt.show()
else:
    print("The weights do not have the expected 3 dimensions.")


#print accuracy on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')





