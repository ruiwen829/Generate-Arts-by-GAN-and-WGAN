## image_resizer.py
# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR 

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...')
np.save('cubism_data.npy', training_data)

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D,LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
import os
from keras.models import load_model

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 1000
# Size vector to generate images from
NOISE_SIZE = 100
# Configuration
EPOCHS = 30000 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/cols
IMAGE_CHANNELS = 3

training_data = np.load('/content/drive/My Drive/dataset/portrait_data.npy')

def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation='relu',input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))
    model.add(UpSampling2D())      # double the size
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation('relu'))
    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding='same'))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation('relu'))
    model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation('tanh'))      # make output to range from (-1,1) as real data
    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)

def build_discriminator(image_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2,
    input_shape=image_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.25))
    
    model.add(Conv2D(512, kernel_size=3, strides=1, padding='same'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(1))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)
    
def save_images(cnt, noise):
    image_array = np.full((
        PREVIEW_MARGIN + (PREVIEW_ROWS * (IMAGE_SIZE + PREVIEW_MARGIN)),
        PREVIEW_MARGIN + (PREVIEW_COLS * (IMAGE_SIZE + PREVIEW_MARGIN)), 3),255, dtype=np.uint8)
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # scale from [-1,1] to [0,1]
    image_count = 0
    for row in range(PREVIEW_ROWS):
        for col in range(PREVIEW_COLS):
            r = row * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            c = col * (IMAGE_SIZE + PREVIEW_MARGIN) + PREVIEW_MARGIN
            image_array[r:r + IMAGE_SIZE, c:c + IMAGE_SIZE] = generated_images[image_count] * 255
            image_count += 1
    output_path = '/content/drive/My Drive/WGAN/Generated Images/output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)
    
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred) # either use -1, 1 for labels, or -mean()
    
image_shape = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
optimizer = RMSprop(lr = 0.00005)

generator = build_generator(NOISE_SIZE, IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)
discriminator.compile(loss=wasserstein,optimizer=optimizer, metrics=['accuracy'])

discriminator.trainable = False


random_input = Input(shape=(NOISE_SIZE,))
generated_image = generator(random_input)
validity = discriminator(generated_image)
combined = Model(random_input, validity)
combined.compile(loss=wasserstein,optimizer=optimizer, metrics=['accuracy'])

valid = np.ones((BATCH_SIZE, 1))
fake = -np.ones((BATCH_SIZE, 1))

discriminator.trainable = True

fixed_noise = np.random.normal(0, 1, (PREVIEW_ROWS * PREVIEW_COLS, NOISE_SIZE))
cnt = 1
clip_threshold = 0.01

for epoch in range(EPOCHS):

    for _ in range(5):
        idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
        x_real = training_data[idx]
 
        noise= np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
        x_fake = generator.predict(noise)
 
        discriminator_metric_real = discriminator.train_on_batch(x_real, valid) #training our discriminator model in both real and fake images separately.

        discriminator_metric_generated = discriminator.train_on_batch(x_fake, fake)
 
        discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

        for l in discriminator.layers:
            weights = l.get_weights()
            weights = [np.clip(w,-clip_threshold, clip_threshold) for w in weights]
            l.set_weights(weights)

    generator_metric = combined.train_on_batch(noise, valid)

    if epoch % SAVE_FREQ == 0:
        save_images(cnt, fixed_noise)
        cnt += 1
 
        print(f'{epoch} epoch, Discriminator accuracy: {100*  discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}')


        discriminator.save('/content/drive/My Drive/WGAN/discriminator.h5')
        generator.save('/content/drive/My Drive/WGAN/generator.h5')
        combined.save('/content/drive/My Drive/WGAN/combined.h5')
