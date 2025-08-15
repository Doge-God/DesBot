import rclpy
from rclpy.node import Node
from des_bot_interfaces.msg import RecFrameResult2

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.lcd.device import st7735
from PIL.ImageDraw import ImageDraw

def draw_centered_ellipse(draw:ImageDraw,x,y,w,h,fill):
    draw.rounded_rectangle([(x-w/2, y-h/2), (x+w/2, y+h/2)],15, fill= fill)

class FaceController(Node):

    def __init__(self):
        super().__init__('face_controller_node')

        serial_0 = spi(port=0, device=0, gpio_DC=6, gpio_RST=5)
        serial_1 = spi(port=1, device=0, gpio_DC=24, gpio_RST=23)

        self.device_0 = st7735(serial_0, rotate=1, gpio_LIGHT=26)
        self.device_1 = st7735(serial_1, rotate=1, gpio_LIGHT=25)

        self.subscription = self.create_subscription(
            RecFrameResult2,
            'rec_frame_result',
            self.rec_frame_callback,
            10
        )


    def __lin_map(self, x, in_min=-1, in_max=1, out_min=0, out_max=160):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    
    def rec_frame_callback(self, msg):
        if not msg.points:
            return

        target = msg.points[0].location

        x = self.__lin_map(target.x, -1,1, 128-20, 20)
        y = self.__lin_map(target.y, -1,1, 35,160-35)

        with canvas(self.device_0) as draw:
            self.device_0.backlight(False)
            draw_centered_ellipse(draw, x, y, 40, 70, 'orange')
        
        with canvas(self.device_1) as draw:
            self.device_1.backlight(False)
            draw_centered_ellipse(draw, x, y, 40, 70, 'orange')

def main(args=None):
    rclpy.init(args=args)
    node = FaceController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

