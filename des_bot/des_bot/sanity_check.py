import serial
import time

# Configuration
port = '/dev/ttyUSB0'
baudrate = 115200 # Adjust if needed
timeout = 1

def open_serial_port(port, baudrate, timeout):
    try:
        ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"Connected to {port} at {baudrate} baud.")
        return ser
    except serial.SerialException as e:
        print(f"Error: {e}")
        return None

def main():
    ser = open_serial_port(port, baudrate, timeout)
    if not ser:
        return

    print("Type commands to send over serial. Type 'exit' to quit.")
    
    try:
        while True:
            user_input = input(">>> ")
            if user_input.lower() == 'exit':
                print("Exiting...")
                break

            # Send command
            ser.write((user_input + '\n').encode())
            time.sleep(0.1)  # Brief pause

            # Read response if available
            response = ser.read_all()
            if response:
                print("Response:", response.decode(errors='ignore'))

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
