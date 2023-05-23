
import serial
import time

# Set up the serial connection
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the appropriate port name

counter = 0

while True:
    # Read data from the serial port
    data = ser.readline().decode().strip()
    # if isinstance(data, str):
    #     print("str")
    print(data)
    pressed = [False for _ in range(6)]

    # # Process the received data
    # if data == 'BUTTON_PRESSED':
    #     # Button is pressed, perform desired action
    #     print("Button pressed!")

    # Add more conditions for different button states if needed
    # if int(data[0]):
    #     counter = 0
    #
    # for bit in data[1:]:
    #     if int(bit):
    #         counter += 1
    # print(counter)
    # time.sleep(0.001)

# Close the serial connection
ser.close()


