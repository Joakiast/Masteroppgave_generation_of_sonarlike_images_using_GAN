"""
python3 -m pip install tensorflow[and-cuda]


"""

import tensorflow as tf

# Function to check available devices
def check_available_devices():
    devices = tf.config.list_physical_devices()
    for device in devices:
        print(device)

    # Check if a GPU is available for TensorFlow
    if tf.config.list_physical_devices('GPU'):
        print('\nDefault GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("\nNo GPU found!")

# Call the function to check for available devices
check_available_devices()