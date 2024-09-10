from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.initializers import GlorotUniform
import numpy as np

def create_model(input_dim, num_classes):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, kernel_initializer=GlorotUniform(), use_bias=True))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(64, kernel_initializer=GlorotUniform(), use_bias=True))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(32, kernel_initializer=GlorotUniform(), use_bias=True))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=GlorotUniform(), use_bias=True))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Example usage
if __name__ == "__main__":
    model = create_model(input_dim=36, num_classes=3)
    # Dummy data for demonstration
    X_train = np.random.random((100, 36))
    y_train = np.random.randint(3, size=(100,))
    y_train = np.eye(3)[y_train]
    model.fit(X_train, y_train, epochs=5)
    model.save('model_final.keras')



