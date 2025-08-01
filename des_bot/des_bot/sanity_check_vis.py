import serial

# Open serial port
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            print(f"Received: {line}")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    ser.close()