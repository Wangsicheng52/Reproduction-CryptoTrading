import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU  # Ensure LeakyReLU is imported
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.dummy import DummyClassifier

class NNModel:

    def __init__(self, in_dim, n_classes, loss='categorical_crossentropy', epochs=32):
        self.dummy_clf = DummyClassifier(strategy="stratified", random_state=2987)
        model = Sequential()
        model.add(Dense(128, activation=LeakyReLU(alpha=0.01), input_dim=in_dim))
        model.add(Dense(64, activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(32, activation=LeakyReLU(alpha=0.01)))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam',
                      loss=[loss],
                      metrics=['categorical_accuracy'])  # Updated metrics
        self.model = model
        self.epochs = epochs

    def train(self, train_data, y):
        encoder = LabelEncoder()
        encoder.fit(y)
        encoded_Y = encoder.transform(y)
        dummy_y = to_categorical(encoded_Y)
        checkpoint = ModelCheckpoint(
            'model_best.keras',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'
        )
        
        history = self.model.fit(
            train_data, 
            dummy_y, 
            batch_size=64, 
            epochs=self.epochs, 
            verbose=1, 
            callbacks=[checkpoint]
        )
        self.model.save('model_final.keras')  # Update to .keras format
        return True

    def predict(self, pred_data):
        y = self.model.predict(pred_data, verbose=0)
        max_data = np.argmax(y, axis=1)
        return max_data  # Adjusted as `max_data - 1` may not be needed

    def load(self, filename):
        custom_objects = {'LeakyReLU': LeakyReLU}
        self.model = load_model(filename, custom_objects=custom_objects)

    def save(self, filename):
        self.model.save(filename)

    def dummy_train(self, train_data, y):
        self.dummy_clf.fit(train_data, y)

    def dummy_predict(self, pred_data):
        preds = self.dummy_clf.predict(pred_data)
        return preds




