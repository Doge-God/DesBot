import rclpy
from rclpy.node import Node
from des_bot_interfaces.msg import RecFrameResult2
import serial

class NaiveTrackingController(Node):

    def __init__(self):
        super().__init__('motor_controller_node')

        # Declare parameters with defaults
        self.declare_parameter('max_x', 3.1)
        self.declare_parameter('min_x', -3.1)
        self.declare_parameter('max_y', 0.45)
        self.declare_parameter('min_y', -0.55)
        self.declare_parameter('deadband_x', 0.05)
        self.declare_parameter('deadband_y', 0.05)
        self.declare_parameter('step_size',0.015)

        # Read parameters
        self.max_x = self.get_parameter('max_x').get_parameter_value().double_value
        self.min_x = self.get_parameter('min_x').get_parameter_value().double_value
        self.max_y = self.get_parameter('max_y').get_parameter_value().double_value
        self.min_y = self.get_parameter('min_y').get_parameter_value().double_value
        self.deadband_x = self.get_parameter('deadband_x').get_parameter_value().double_value
        self.deadband_y = self.get_parameter('deadband_y').get_parameter_value().double_value
        self.step_size = self.get_parameter('step_size').get_parameter_value().double_value

        # Motor positions in radians
        self.motor_x = 0.0
        self.motor_y = 0.0

        # Setup serial connection
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info('Serial connection established.')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial connection failed: {e}')
            self.serial_port = None

        # Subscription to recognition frame result
        self.subscription = self.create_subscription(
            RecFrameResult2,
            'rec_frame_result',
            self.rec_frame_callback,
            10
        )

    def rec_frame_callback(self, msg):
        if not msg.points:
            return

        target = msg.points[0].location

        # Motor X logic
        if target.x > self.deadband_x:
            self.motor_x += self.step_size
        elif target.x < -self.deadband_x:
            self.motor_x -= self.step_size

        # Motor Y logic
        if target.y > self.deadband_y:
            self.motor_y += self.step_size
        elif target.y < -self.deadband_y:
            self.motor_y -= self.step_size

        # Clip and warn
        clipped = False
        if self.motor_x > self.max_x:
            self.motor_x = self.max_x
            self.get_logger().warn('motor_x clipped to max_x')
            clipped = True
        elif self.motor_x < self.min_x:
            self.motor_x = self.min_x
            self.get_logger().warn('motor_x clipped to min_x')
            clipped = True

        if self.motor_y > self.max_y:
            self.motor_y = self.max_y
            self.get_logger().warn('motor_y clipped to max_y')
            clipped = True
        elif self.motor_y < self.min_y:
            self.motor_y = self.min_y
            self.get_logger().warn('motor_y clipped to min_y')
            clipped = True

        # Send motor commands
        self.send_motor_command('A', self.motor_x)
        self.send_motor_command('B', self.motor_y)

    def send_motor_command(self, motor_id, angle):
        if self.serial_port is not None:
            command = f"{motor_id}{angle:.3f}\n"
            try:
                self.serial_port.write(command.encode())
                self.get_logger().info(f'Sent command: {command.strip()}')
            except serial.SerialException as e:
                self.get_logger().error(f'Failed to send command: {e}')
        else:
            self.get_logger().warn(f'Skipped command {motor_id}{angle:.3f} - No serial connection.')

def main(args=None):
    rclpy.init(args=args)
    node = NaiveTrackingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
