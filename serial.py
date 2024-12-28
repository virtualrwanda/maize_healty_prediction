import sys
# Temporarily remove the current directory from the sys.path
current_directory = sys.path[0]
sys.path.pop(0)

import serial  # Now it will import the pyserial library

# Add the current directory back to sys.path
sys.path.insert(0, current_directory)

import time

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyUSB0',  # Replace with your serial port name (e.g., 'COM3' on Windows)
    baudrate=9600,        # Set the baud rate according to your device's configuration
    timeout=1             # Timeout for read operations
)

# Give the connection a second to establish
time.sleep(2)

# Read data from serial port
try:
    while True:
        if ser.in_waiting > 0:  # Check if there's data to read
            data = ser.readline().decode('utf-8').rstrip()  # Read a line of data, decode it, and remove any trailing newline
            print("Received data:", data)
except KeyboardInterrupt:
    print("Exiting program.")

finally:
    ser.close()  # Close the serial connection when done
