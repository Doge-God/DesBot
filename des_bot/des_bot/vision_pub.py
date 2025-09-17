import rclpy
from rclpy.node import Node
import serial
import threading
from des_bot_interfaces.msg import RecFrameResult2, RecPoint2, Vec2

class VisionResultPublisher(Node):
    def __init__(self):
        super().__init__('rec_frame_result_publisher')


        # Declare max rate parameter
        self.declare_parameter('max_publish_rate', 40.0)  # Hz
        self.declare_parameter('screen_w', 640)  # Hz
        self.declare_parameter('screen_h', 480)  # Hz

        self.max_rate = self.get_parameter('max_publish_rate').get_parameter_value().double_value
        self.screen_w = self.get_parameter('screen_w').get_parameter_value().integer_value
        self.screen_h = self.get_parameter('screen_h').get_parameter_value().integer_value

        self.min_interval = 1.0 / self.max_rate

        # Initialize last publish time
        self.last_publish_time = self.get_clock().now()
        
        # Create publisher
        self.publisher_ = self.create_publisher(RecFrameResult2, 'rec_frame_result', 10)
        
        # Setup serial
        self.ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

        # Start reader thread
        self.running = True
        self.thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
        self.thread.start()

    def serial_reader_loop(self):
        while self.running and rclpy.ok():
            try:
                line = self.ser.readline().decode('utf-8').strip()
                if line.startswith('#') and line.endswith('$'):
                    # check for maximum publishing rate
                    now = self.get_clock().now()
                    if (now - self.last_publish_time).nanoseconds < int(self.min_interval * 1e9):
                        continue  # Skip this message due to rate limiting

                    content = line[1:-1]  # Remove '#' and '$'
                    detections = content.split(';')

                    msg = RecFrameResult2()
                    points = []
                    for det in detections:
                        try:
                            x, y, w, h = map(int, det.split(','))
                            point = RecPoint2()
                            point.location = Vec2(
                                # x = float(x),
                                # y = float(y)
                                x=float( (x-self.screen_w/2+w/2)/ (self.screen_w/2) ), 
                                y=float( (y-self.screen_h/2+h/2)/ (self.screen_h/2) )
                            )
                            point.score = float( max(w, h) )
                            points.append(point)
                        except ValueError as e:
                            self.get_logger().warn(f"Failed to parse detection '{det}': {e}")

                    # sort points by size of bounding box (roughly), descending
                    points.sort(key=lambda p: p.score, reverse=True)
                    msg.points = points

                    self.publisher_.publish(msg)
                    self.last_publish_time = now
                    # self.get_logger().info(f"Published {len(msg.points)} points")

            except Exception as e:
                self.get_logger().error(f"Serial read error: {e}")

    def destroy_node(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisionResultPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()
