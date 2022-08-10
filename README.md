# The Research of Online Federated Transfer Learning with Microcontrollers

# Create dataset
- [Edge Impulse](https://www.edgeimpulse.com/)
- [Connecting to Edge Impulse](https://docs.edgeimpulse.com/docs/development-platforms/officially-supported-mcu-targets/arduino-nano-33-ble-sense#4.-verifying-that-the-device-is-connected)
- [Data acquisition](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition)

# How to use it
- Configure the Arduino Nano 33 BLE Sense boards like in the image.

![](https://i.imgur.com/7AF9jaF.png)

- Open the project with PlatformIO and flash the firmware to all the boards.
- Run the ```fl_server.py``` using Python3
    - Specify the number of devices used
    - Specify the Serial ports of each device
- Start training the devices using the buttons.
    - The 3 buttons on the middle are used to train 3 different keywords (to be decided by you!)
    - The board will start recording when the button is pressed & RELEASED (one second).
    - The fourth button can be used to start the Federated Learning process. It can be configured on the main loop in src/main.ino.
    - The top button can be used to only run the inference without training.