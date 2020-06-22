"""
Title: GANdinsky
Description: Creating art from portrait and abstract datasets
Course: Neural Networks and Deep Learning
Instructor: Dr. Farid Alizadeh
Group: C
Team Members: Elnaz A. Torkamani, Ruiwen Zhang, Sunit Nair
@author: groupc.nndl
"""

#The following commented line must be run in a new Colab session before running the actual code
%tensorflow_version 1.x

from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from google.colab import files
from google.colab import drive
import time
import matplotlib.pyplot as plt

print("Starting process.")
print ("Current date and time:",time.strftime("%Y-%m-%d %H:%M:%S"))

drive.mount('/content/drive', force_remount=True)

dir_ext=time.strftime('%Y%m%d%H%M%S')
print("Creating output folder on colab with name:",dir_ext)
drive_path='drive/My Drive/colab_output/'+dir_ext
if not os.path.exists(drive_path):
        os.makedirs(drive_path)

# Grid to view generated images
PREVIEW_ROWS = 4
PREVIEW_COLS = 6
PREVIEW_MARGIN = 4

#Other hyperparameters
SAVE_FREQ = 100 #Save images every n iterations
NOISE_SIZE = 128 #Starting point for image generation
EPOCHS = 50000 #Total number of iterations
BATCH_SIZE = 64 #Batch size for epoch
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3
CONTINUE_TRAINING = False   #Continue training with saved models (change path where models are saved to ensure)
DATASET_VAR = 'portrait'  #Dataset to use (portrait/abstract)

#Ensure that Google Drive is munted and file is available at path (or change path as required)
#Reading data from .npy file containing compressed image data (128 x 128) stored on Google Drive
training_data=np.load('drive/My Drive/prepared/'+DATASET_VAR+'_data.npy')

#Build discriminator model
def build_discriminator(image_shape):    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))    
    
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))    
    
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)

#Build generator model
def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding="same"))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation("relu"))
         
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    
    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)

#Function to save images at every SAVE_FREQ iteration
def save_images(cnt, noise):
    image_array = np.full((PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)), PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3), 255, dtype=np.uint8)

    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'drive/My Drive/colab_output/'+dir_ext+'/images'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

#Function to save models after completion
def save_models(discriminator, generator, combined):
    print("Creating folder to save models...")
    model_path = 'drive/My Drive/colab_output/'+dir_ext+'/models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    d_filename = os.path.join(model_path,'discriminator.h5')
    g_filename = os.path.join(model_path,'generator.h5')
    c_filename = os.path.join(model_path,'combined.h5')
    discriminator.reset_metrics()
    generator.reset_metrics()
    combined.reset_metrics()
    print("Saving discriminator model...")
    discriminator.save(d_filename)
    print("Saving generator model...")
    generator.save(g_filename)
    print("Saving combined model...")
    combined.save(c_filename)
    
def plot_history(c1_loss, c1_acc, c2_loss, c2_acc, c3_loss, c3_acc, g_loss, g_acc):
	  #plot loss and accuracy history
    print("Creating folder to save plots...")
    plot_path = 'drive/My Drive/colab_output/'+dir_ext+'/plots'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    loss_filename=os.path.join(plot_path,'loss_plot.png')
    acc_filename=os.path.join(plot_path,'acc_plot.png')
    plt.plot(c1_loss, label='discriminator_real')
    plt.plot(c2_loss, label='discriminator_fake')
    plt.plot(c3_loss, label='discriminator_average')
    plt.plot(g_loss, label='generator')
    plt.legend(loc="best")
    plt.title('Loss')
    print("Saving loss plot...")
    plt.savefig(loss_filename)
    plt.close()
    plt.plot(c1_acc, label='discriminator_real')
    plt.plot(c2_acc, label='discriminator_fake')
    plt.plot(c3_acc, label='discriminator_average')
    plt.plot(g_acc, label='generator')
    plt.legend(loc="best")
    plt.title('Accuracy')
    print("Saving accuracy plot...")
    plt.savefig(acc_filename)
    plt.close()

image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = Adam(1.5e-4, 0.5)
discriminator = build_discriminator(image_shape)
if CONTINUE_TRAINING:
    discriminator.load_weights('drive/My Drive/models/'+DATASET_VAR+'/discriminator.h5')
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer, metrics=["accuracy"])
generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)
if CONTINUE_TRAINING:
    generator.load_weights('drive/My Drive/models/'+DATASET_VAR+'/generator.h5')
random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)

#Ensures that the generator fits weights to better learn from dataset and not just to beat the discriminator
discriminator.trainable = False
validity = discriminator(generated_image)

combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy",optimizer=optimizer, metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))

c1_loss, c1_acc, c2_loss, c2_acc, c3_loss, c3_acc, g_loss, g_acc = list(), list(), list(), list(), list(), list(), list(), list()

cnt = 0
for epoch in range(EPOCHS+1):
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    x_real = training_data[idx]
 
    noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
    x_fake = generator.predict(noise)
 
    discriminator_metric_real = discriminator.train_on_batch(x_real, y_real)
    discriminator_metric_generated = discriminator.train_on_batch(x_fake, y_fake)
 
    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)
    generator_metric = combined.train_on_batch(noise, y_real)

    c1_loss.append(discriminator_metric_real[0])
    c2_loss.append(discriminator_metric_generated[0])
    c3_loss.append(discriminator_metric[0])
    g_loss.append(generator_metric[0])
    c1_acc.append(discriminator_metric_real[1])
    c2_acc.append(discriminator_metric_generated[1])
    c3_acc.append(discriminator_metric[1])
    g_acc.append(generator_metric[1])

    if (epoch % SAVE_FREQ == 0) or (epoch == 0):
        save_images(cnt, fixed_noise)
        cnt += 1
        print(f"Epoch {epoch}:  Discriminator accuracy (train/test): {discriminator_metric_real} / {discriminator_metric_generated} Generator accuracy: {generator_metric}")

save_models(discriminator, generator, combined)
plot_history(c1_loss, c1_acc, c2_loss, c2_acc, c3_loss, c3_acc, g_loss, g_acc)
print ("Current date and time:",time.strftime("%Y-%m-%d %H:%M:%S"))
print("End of program. Process completed successfully.")
