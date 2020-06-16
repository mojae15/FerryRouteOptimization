# Based on: https://www.tensorflow.org/tutorials/keras/regression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import os
import itertools

# I get some annoying warnings, so suppress them
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
        
    import tensorflow as tf

    from tensorflow import keras
    from tensorflow.keras import layers

# print(tf.__version__);

# Build Model

def build_model(optimizer, activation_function, hidden_layers, dropout_locations, dropout_rate):
    
    model = Sequential()
    model.add(Dense(64, kernel_initializer='normal', activation=activation_function, input_shape=[len(train_dataset.keys())]))
    
    for i in range(hidden_layers):
        model.add(Dense(64, kernel_initializer='normal', activation=activation_function))
        if (i in dropout_locations):
            model.add(Dropout(dropout_rate))


    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    # model = keras.Sequential([
    #     layers.Dense(64, kernel_initializer='normal', activation=activation_function, input_shape=[len(train_dataset.keys())]),
    #     # layers.Dropout(0.2),
    #     layers.Dense(64, kernel_initializer='normal', activation=activation_function),
    #     # layers.Dropout(0.2),
    #     # layers.Dense(64, kernel_initializer='normal', activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=0.1)),
    #     # layers.Dropout(0.2),
    #     # layers.Dense(64, kernel_initializer='normal', activation=activation_function, kernel_regularizer=keras.regularizers.l2(l=0.1)),
    #     # layers.Dropout(0.2),
    #     # layers.Dense(64, kernel_initializer='normal',activation=activation_function),
    #     # layers.Dropout(0.2),
    #     # layers.Dense(64, kernel_initializer='normal', activation=activation_function),
    #     # layers.Dropout(0.2),
    #     # layers.Dense(64, kernel_initializer='normal', activation=activation_function),
    #     # layers.Dropout(0.5),
    #     layers.Dense(1, kernel_initializer='normal', activation='linear')
    # ])

    if (str(optimizer) == 'Adam'):
        opti =  tf.keras.optimizers.Adam(learning_rate=0.001)
    else:
        opti = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=opti, metrics=['mae', 'mse'])
    return model

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [fuel_consumption]')

    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()
    plt.savefig(dt_string+'/history_mea.png')
    plt.savefig(dt_string+'/history_mea.pdf')
    # plt.clf()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$fuel\_consumption^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')

    # plt.ylim([0,20])
    plt.legend()
    plt.savefig(dt_string+'/history_mse.png')
    plt.savefig(dt_string+'/history_mse.pdf')
    plt.clf()

EPOCHS = 1000

raw_dataset = pd.read_csv('relative_data.csv', na_values="?", comment= '\t', skipinitialspace=True)

# raw_dataset = raw_dataset[['Actual RPM Thruster 1 RPM', 'Actual RPM Thruster 2 RPM', 'Actual RPM Thruster 3 RPM', 'Actual RPM Thruster 4 RPM', 'DPT - Depth m', 'Heading True °', 'current_magnitude', 'current_direction', 'wind_speed', 'wind_direction', 'speeds', 'fuel_consumption']]

print(raw_dataset)

optimizers = ['Adam', 'RMSProp']
activation_functions = ['tanh', 'sigmoid', 'relu']
dropout_rates = [0.4]


dt_string = ""

##### PREPARE DATASET #####

dataset = raw_dataset.copy()
# print(dataset.tail())


# Check for NaN, should be no such values
# print(dataset.isna().sum())



print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Visualise some of the data
# sns.pairplot(train_dataset[list(train_dataset)], diag_kind='kde')
# plt.savefig(dt_string+'/pairplot.png')
# plt.clf()


# Show some stats
train_stats = train_dataset.describe()
train_stats.pop('fuel_consumption')
train_stats = train_stats.transpose()

# print(train_stats)


# Split features from labels
train_labels = train_dataset.pop('fuel_consumption')
test_labels = test_dataset.pop('fuel_consumption')

# Normalise data
def norm(x):
    return (x -train_stats['mean'] / train_stats['std'])

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
c = 0

for opti in optimizers:
    for af in activation_functions:
        for dr in dropout_rates:
            hidden_layers = 1
            while hidden_layers < 5:
                for L in range(0, hidden_layers+1):
                    for dropout_locations in itertools.combinations(range(hidden_layers), L):
                        
                        c = c +1

                        if (c > 176):
                            
                            now = datetime.now()

                            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

                            dt_string = "Relative/"+dt_string

                            if not os.path.exists(dt_string):
                                os.makedirs(dt_string)
                            

                            raw_dataset.to_csv(dt_string+'/data.csv',index=None)


                            model = build_model(opti, af, hidden_layers, dropout_locations, dr)
                            
                            # # Early stopping
                            # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

                            model_checkpoint = keras.callbacks.ModelCheckpoint(dt_string + '/model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

                            history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split=0.25, verbose=0, callbacks=[model_checkpoint])

                            tf.keras.utils.plot_model(
                                model,
                                to_file=dt_string+'/model.png',
                                show_shapes=False,
                                show_layer_names=True
                            )

                            # hist = pd.DataFrame(history.history)
                            # hist['epoch'] = history.epoch
                            # print(hist.tail())

                            plot_history(history)

                            loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

                            print("Testing set Mean Abs Error: {:5.2f} fuel_consumption".format(mae))

                            test_predictions = model.predict(normed_test_data).flatten()

                            plt.scatter(test_labels, test_predictions, s=3)
                            plt.xlabel('True Values [fuel_consumption]')
                            plt.ylabel('Predictions [fuel_consumption]')

                            xmin = plt.xlim()[0]-0.3
                            xmax = plt.xlim()[1]+0.3

                            ymin = plt.ylim()[0]-0.3
                            ymax = plt.ylim()[1]+0.3

                            plt.axis('equal')
                            plt.axis('square')
                            plt.xlim([0,xmax])
                            plt.ylim([0,ymax])
                            _ = plt.plot(plt.xlim(),plt.ylim())

                            plt.xlim(xmin, xmax)
                            plt.ylim(ymin, ymax)

                            plt.savefig(dt_string+'/predictions.png')
                            plt.savefig(dt_string+'/predictions.pdf')
                            plt.clf()

                            error = test_predictions - test_labels
                            plt.hist(error, bins = 25)
                            plt.xlabel("Prediction Error [fuel_consumption]")
                            _ = plt.ylabel("Count")
                            plt.savefig(dt_string+'/error.png')
                            plt.savefig(dt_string+'/error.pdf')

                            plt.clf()



                            # Write file with the timestamp as name that describes the input features used and how the network looks


                            f = open(dt_string+"/model.txt", 'w+')

                            f.write("Features used:\n")
                            for col in raw_dataset.columns:
                                f.write("\t" + col +"\n")

                            f.write("Mean Squared Error: {:5.4f}".format(mse))
                            f.write("\n")
                            f.write("Mean Abs Error: {:5.4f}".format(mae))
                            f.write("\n")
                            f.write("Loss: {:5.4f}".format(loss))
                            f.write("\n")

                            f.write("\n")
                            f.write("Model: ")
                            model.summary(print_fn=lambda x: f.write(x + '\n'))

                            f.write("\n")
                            f.write("Optimizer: "+str(opti))
                            f.write("\n")
                            f.write("Activation function: "+af)
                            f.write("\n")
                            f.write("Dropout Rate: " + str(dr))
                            f.write("\n")
                            f.write("Hidden Layers: " + str(hidden_layers))
                            f.write("\n")
                            f.write("Dropout Locations: ")
                            f.write(' '.join(str(s) for s in dropout_locations))
                            f.write("\n\n")
                            f.close()
                            plt.close('all')
                        else:
                            model = build_model(opti, af, hidden_layers, dropout_locations, dr)





                hidden_layers = hidden_layers +1


exit()











#TODO
# - Create proper data file that can be imported to it's easy to work with with keras, the things I've tried so far did not look pretty, so they have been removed
# - After that is doone it should be relatively easy to train the model



