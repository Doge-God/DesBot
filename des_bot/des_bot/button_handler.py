import rclpy
from rclpy.node import Node
from gpiozero import Button, LED
from des_bot_interfaces.srv import StartConversation, EndConversation

class ButtonHandler(Node):
    def __init__(self):
        super().__init__('button_handler')
        
        # Create service clients
        self.start_conv_client = self.create_client(StartConversation, 'start_conversation')
        self.end_conv_client = self.create_client(EndConversation, 'end_conversation')
        
        # Initialize buttons with GPIO pins
        self.main_button = Button(14, bounce_time=0.05)
        self.stop_button = Button(22, bounce_time=0.05)
        
        #dummy sink
        self.out1 = LED(15)
        self.out2 = LED(27)
        
        # Set button callbacks
        self.main_button.when_pressed = self.on_main_button_pressed
        self.stop_button.when_pressed = self.on_stop_button_pressed
        
        self.get_logger().info('Button handler initialized')

    def on_main_button_pressed(self):
        self.get_logger().info('Main button pressed - Starting conversation')
        self.call_start_conversation()

    def on_stop_button_pressed(self):
        self.get_logger().info('Stop button pressed - Ending conversation')
        self.call_end_conversation()

    def call_start_conversation(self):
        if not self.start_conv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Start conversation service not available')
            return

        request = StartConversation.Request()
        future = self.start_conv_client.call_async(request)
        future.add_done_callback(self.start_conversation_callback)

    def call_end_conversation(self):
        if not self.end_conv_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('End conversation service not available')
            return

        request = EndConversation.Request()
        future = self.end_conv_client.call_async(request)
        future.add_done_callback(self.end_conversation_callback)

    def start_conversation_callback(self, future):
        try:
            response = future.result()
            if response.is_successful:
                self.get_logger().info('Successfully started conversation')
            else:
                self.get_logger().warning('Failed to start conversation')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

    def end_conversation_callback(self, future):
        try:
            response = future.result()
            if response.is_successful:
                self.get_logger().info('Successfully ended conversation')
            else:
                self.get_logger().warning('Failed to end conversation')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ButtonHandler()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()