import asyncio
import threading
from typing import Optional
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import serial
from des_bot_interfaces.msg import RecFrameResult2
from std_msgs.msg import String

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.lcd.device import st7735
from PIL.ImageDraw import ImageDraw
from PIL import ImageFont
from queue import Queue, Full
from gpiozero import PWMLED

from .utils.conversation_state_types import ConversationState, ExpressionState, SpeakingEmote
from .utils.draw import draw_centered_ellipse, draw_centered_ellipse_top_half, draw_centered_rectangle
# 

class ExpressionManager():


    def __init__(self, node:Node):
        
        self.node = node
        
     

        #==== Eye setup ----------------------------------------------
        serial_0 = spi(port=0, device=0, gpio_DC=6, gpio_RST=5)
        serial_1 = spi(port=1, device=0, gpio_DC=24, gpio_RST=23)

        self.device_0 = st7735(serial_0, rotate=1, gpio_LIGHT=26)
        self.device_1 = st7735(serial_1, rotate=1, gpio_LIGHT=25)
        self.backlight_on()
        # eye position: eye pos track target pos, updated with timer
        self.eye_pos = [int(self.__lin_map(0, -1,1, 128-35, 35)),(self.__lin_map(0, -1,1, 35,160-35))]
        self.eye_target_pos =  [int(self.__lin_map(0, -1,1, 128-20, 20)),int(self.__lin_map(0, -1,1, 35,160-35))]
        # task
        self.eye_task:Optional[asyncio.Task]= None
        
        # ---- Button LED setup -----------------------------------------------------
        self.antenna_led = PWMLED(2)
        self.antenna_led_value = 0.0
        self.main_button_led = PWMLED(3)
        self.main_button_led_value = 0.0

        # ---- Motor setup -----------------------------------------------------
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.node.get_logger().info('Serial connection established.')
        except serial.SerialException as e:
            self.node.get_logger().error(f'Serial connection failed: {e}')
            self.serial_port = None
        # limits -----------------
        self.max_a = 1
        self.min_a = -2.2
        self.max_b = 1.68
        self.min_b = -1.5
        # target -----------------
        self.a_target = 0.0


        # Face refresh setup
        self.eye_pos_update_timer = self.node.create_timer(
            0.0368, 
            self.track_target_pos_callback)

        # Rec result refresh
        self.rec_sub = None
        self.rec_queue = Queue(5)
        # [[self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]]

        self.expression_state_sub = self.node.create_subscription(
            String,
            'expression_state',
            self.state_callback,
            10
        )   

        self.last_state = ExpressionState.IDLE
        
        self.target_pos = [0,0]

        self.center_eye_pos = [self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]
        self.eye_target_pos =  [self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]

    def clean_up(self):
        if self.eye_task:
            self.eye_task.cancel()
        self.send_motor_command('A', 0.0)
        self.send_motor_command('B', 0.0)
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
    ################################################################################################
    ############## CALLBACKS ############################################
    def track_target_pos_callback(self):
        '''Both eye_pos and motor_a_pos track their target value by one step'''
        pass
    
    def state_callback(self, msg):
        # Convert string back to ConversationState flag
        try:
            state_flag = ExpressionState[msg.data]
            self.react_to_state_change(state_flag)
            self.last_state = state_flag
        except KeyError:
            self.node.get_logger().warn(f"Unknown state received: {msg.data}")

    def rec_frame_callback(self, msg):
        if not msg.points:
            return
        target = msg.points[0].location
        x = self.__lin_map(target.x, -1,1, 128-20, 20)
        y = self.__lin_map(target.y, -1,1, 35,160-35)

        self.target_pos = [x,y]

    ## Handle State change:
    def react_to_state_change(self, new_state:ExpressionState):
        if new_state == ExpressionState.LISTENING and self.last_state == ExpressionState.IDLE:
            # Switch to listening eyes

            if self.eye_task:
                self.eye_task.cancel()
            self.eye_task = asyncio.create_task(self.run_eye_neutral())

    ################################################################################################
    ############## Eye tasks ############################################
    async def run_eye_neutral(self):
        print(">>> Eye neutral task started.")
        for _ in range(2):
            with canvas(self.device_0) as dev0_draw, canvas(self.device_1) as dev1_draw:
                self.draw_blink_eyes(dev0_draw, dev1_draw, self.eye_pos[0], self.eye_pos[1])
            await asyncio.sleep(0.05)
        for _ in range(30):
            with canvas(self.device_0) as dev0_draw, canvas(self.device_1) as dev1_draw:
                self.draw_neutral_eyes(dev0_draw, dev1_draw, self.eye_pos[0], self.eye_pos[1])
            await asyncio.sleep(0.05)

        self.eye_task = asyncio.create_task(self.run_eye_neutral())
    ################################################################################################
    ############## Eye Draw ############################################
    def backlight_on(self):
        self.device_0.backlight(False)
        self.device_1.backlight(False)

    def draw_blink_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw, x, y):
        draw_centered_rectangle(dev0_draw, x, y, 70, 5, '#ffc300', 2)
        draw_centered_rectangle(dev1_draw, x, y, 70, 5, '#ffc300', 2)
    
    def draw_neutral_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw, x, y):
        draw_centered_ellipse(dev0_draw, x, y, 70, 70, '#ffc300')
        draw_centered_ellipse(dev1_draw, x, y, 70, 70, '#ffc300')

    def draw_thinking_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw, x, y):
        draw_centered_rectangle(dev0_draw, x, y, 70, 25, '#ffc300', 12)  
        draw_centered_rectangle(dev1_draw, x, y, 70, 25, '#ffc300', 12)
        
    ################################################################################################
    ############## UTIL ############################################
    def __lin_map(self, x, in_min=-1, in_max=1, out_min=0, out_max=160):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def draw_centered_text(self, draw:ImageDraw, text_to_draw, x, y, font_size=15, font_path=None, fill="white"):
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default(font_size)
        text_width = draw.textlength(text_to_draw, font=font)

      
        text_x = int(x - text_width // 2)
        text_y = y
        draw.text((text_x, text_y), text_to_draw, font=font, fill=fill)
    


rclpy.init()
node = Node("expression_manager")
expression_manager = ExpressionManager(node)


async def ros_loop():

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0)
        await asyncio.sleep(1e-4)
    expression_manager.shutdown()

def main():
    print("node started")
    asyncio.run(ros_loop())

if __name__ == '__main__':
    main()
# ##

# if __name__ == '__main__':
#     main()

# def main(args=None):
#     rclpy.init(args=args)
#     node = ExpressionManager()
#     try:
#         rclpy.spin(node)
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

