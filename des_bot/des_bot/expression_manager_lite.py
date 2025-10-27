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

class ExpressionManagerLite(Node):


    def __init__(self):
        
        super().__init__('expression_manager')

        #==== Eye setup ----------------------------------------------
        serial_0 = spi(port=0, device=0, gpio_DC=6, gpio_RST=5)
        serial_1 = spi(port=1, device=0, gpio_DC=24, gpio_RST=23)

        self.device_0 = st7735(serial_0, rotate=1, gpio_LIGHT=26)
        self.device_1 = st7735(serial_1, rotate=1, gpio_LIGHT=25)
        self.backlight_on()
        # eye position: eye pos track target pos, updated with timer
        self.eye_pos = [int(self.__lin_map(0, -1,1, 128-35, 35)),(self.__lin_map(0, -1,1, 35,160-35))]
        # self.eye_target_pos =  [int(self.__lin_map(0, -1,1, 128-35, 35)),int(self.__lin_map(0, -1,1, 35,160-35))]
        
        # ---- Button LED setup -----------------------------------------------------
        self.antenna_led = PWMLED(2)
        self.antenna_led_value = 0.0
        self.main_button_led = PWMLED(3)
        self.main_button_led_value = 0.0

        # ---- Motor setup -----------------------------------------------------
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info('Serial connection established.')
        except serial.SerialException as e:
            self.get_logger().error(f'Serial connection failed: {e}')
            self.serial_port = None
        # limits -----------------
        self.max_a = 1
        self.min_a = -2.2
        self.max_b = 1.68
        self.min_b = -1.5
        # target -----------------
        self.a_target = 0.0
        self.b_target = 0.0
        self.a_deadzone = 0.15
        self.step_size_a = 0.02

        #==========================================================================

        # Face refresh setup
        self.tick_timer = self.create_timer(
            0.0368, 
            self.tick_callback)

        # Rec result refresh
        self.rec_sub = None
        self.rec_queue = Queue(5)
        # [[self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]]

        self.expression_state_sub = self.create_subscription(
            String,
            'expression_state',
            self.state_callback,
            10
        )   
        self.rec_sub = self.create_subscription(
            RecFrameResult2,
            'rec_frame_result',
            self.rec_frame_callback,
            10
        )
        
        self.current_state = ExpressionState.IDLE
        self.last_state = ExpressionState.IDLE
        
        self.target_pos = [0,0]
        self.blink_counter = 0
        self.blink_interval = 90  # ticks


    def cleanup(self):

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
    ################################################################################################
    ############## CALLBACKS ############################################
    def tick_callback(self):
        '''Both eye_pos and motor_a_pos track their target value by one step'''
        eye_x, eye_y = self.calc_eye_pos(self.target_pos[0], self.target_pos[1])

        self.blink_counter += 1

        with canvas(self.device_0) as dev0_draw, canvas(self.device_1) as dev1_draw:
            if self.blink_counter >= self.blink_interval:
                self.draw_blink_eyes(dev0_draw, dev1_draw, eye_x, eye_y)
                if self.blink_counter >= self.blink_interval + 1:  # blink duration
                    self.blink_counter = 0
            
            elif self.current_state == ExpressionState.LISTENING:
                self.draw_neutral_eyes(dev0_draw, dev1_draw, eye_x, eye_y)
            elif self.current_state == ExpressionState.GENERATING:
                self.draw_thinking_eyes(dev0_draw, dev1_draw, eye_x, eye_y)
            elif self.current_state == ExpressionState.SPEAKING:
                self.draw_happy_eyes(dev0_draw, dev1_draw, eye_x, eye_y)
    
    def state_callback(self, msg):
        # Convert string back to ConversationState flag
        try:
            state_flag = ExpressionState[msg.data]
            self.react_to_state_change(state_flag)
            self.last_state = state_flag
        except KeyError:
            self.get_logger().warn(f"Unknown state received: {msg.data}")

    def rec_frame_callback(self, msg):
        if not msg.points:
            return
        target = msg.points[0].location
        self.target_pos = [target.x,target.y]

        if self.current_state != ExpressionState.IDLE:
            self.motor_tick()

    ## Handle State change: *********************************************************************************************
    def react_to_state_change(self, new_state:ExpressionState):
        if new_state == ExpressionState.LISTENING and self.last_state == ExpressionState.IDLE:
            self.send_motor_command('A',0.6)
            self.send_motor_command('B', 0)
    

        if new_state == ExpressionState.IDLE:
            self.send_motor_command('A',-2.2)
            self.send_motor_command('B',1.68)
            self.target_pos = [0,0]
            self.antenna_led.off()
            self.main_button_led.off()

        if new_state == ExpressionState.GENERATING and self.last_state == ExpressionState.LISTENING:
            self.main_button_led.pulse(fade_in_time=0.3, fade_out_time=0.5)    
        
        if new_state == ExpressionState.GENERATING:
            self.main_button_led.pulse(fade_in_time=0.3, fade_out_time=0.5)
            self.antenna_led.off()
        if new_state == ExpressionState.SPEAKING:
            self.antenna_led.value = 0
            self.main_button_led.value = 1
        if new_state == ExpressionState.LISTENING:
            self.antenna_led.value = 1
            self.main_button_led.value = 0

        self.last_state = self.current_state
        self.current_state = new_state

    def motor_tick(self):
        # Motor X logic
        if self.target_pos[1] > self.a_deadzone:
            self.a_target -= self.step_size_a
        elif self.target_pos[1] < -self.a_deadzone:
            self.a_target += self.step_size_a

        # Clip and warn
        if self.a_target > self.max_a:       
            self.a_target = self.max_a
            # self.get_logger().warn('motor_x clipped to max_x')
        elif self.a_target < self.min_a:
            self.a_target = self.min_a
            # self.get_logger().warn('motor_x clipped to min_x')   
        
        self.b_target = -self.a_target
        # Clip and warn
        if self.b_target > self.max_b:       
            self.b_target = self.max_b
            # self.get_logger().warn('motor_x clipped to max_x')
        elif self.b_target < self.min_b:
            self.b_target = self.min_b
            # self.get_logger().warn('motor_x clipped to min_x')
        self.send_motor_command('A', self.a_target)      

        if self.current_state == ExpressionState.LISTENING:      
            self.send_motor_command('B', self.b_target)   
        else:              
            self.send_motor_command('B', 1.68)
    ################################################################################################
    ############## Eye Draw ############################################
    def backlight_on(self):
        self.device_0.backlight(False)
        self.device_1.backlight(False)

    def draw_blink_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw, x, y):
        draw_centered_rectangle(dev0_draw, x, y, 70, 5, '#ffc300', 2)
        draw_centered_rectangle(dev1_draw, x, y, 70, 5, '#ffc300', 2)
    
    def draw_happy_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw, x, y):
        draw_centered_ellipse_top_half(dev0_draw, x, y, 70, 70, '#ffc300',0.75)
        draw_centered_ellipse_top_half(dev1_draw, x, y, 70, 70, '#ffc300',0.75)
    
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

    def calc_eye_pos(self, target_x, target_y):
        eye_x = int(self.__lin_map(target_x, -1,1, 128-35, 35))
        eye_y = int(self.__lin_map(target_y, -1,1, 35,160-35))
        return (eye_x, eye_y)

    def draw_centered_text(self, draw:ImageDraw, text_to_draw, x, y, font_size=15, font_path=None, fill="white"):
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default(font_size)
        text_width = draw.textlength(text_to_draw, font=font)

      
        text_x = int(x - text_width // 2)
        text_y = y
        draw.text((text_x, text_y), text_to_draw, font=font, fill=fill)

    def send_motor_command(self, motor_id, angle):
        if self.serial_port is not None:
            command = f"{motor_id}{angle:.3f}\n"
            try:
                self.serial_port.write(command.encode())
                # self.get_logger().info(f'Sent command: {command.strip()}')
            except serial.SerialException as e:
                self.get_logger().error(f'Failed to send command: {e}')
        else:
            self.get_logger().warn(f'Skipped command {motor_id}{angle:.3f} - No serial connection.')



def main(args=None):
    rclpy.init(args=args)
    node = ExpressionManagerLite()
    try:
        rclpy.spin(node)
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()