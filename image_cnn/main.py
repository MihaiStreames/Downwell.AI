# Convolutional Neural Networks (CNNs) will be used to spot enemies, walls, and other objects in the game
# The CNN will be trained on a set of screenshots of the game

# Hitbox around player - have to find a way to ignore the gunshots and the bullets, as well as any debris
# It should return the type of enemies spotted and how far away they are - potentially add hitboxes around them
# It should also return the type of walls / platforms, because some walls / platforms can damage the player
# It should also return the type of powerups, which are only found in the shops and side rooms


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator