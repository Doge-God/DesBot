import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from des_bot_interfaces.msg import RecFrameResult2
from std_msgs.msg import String

from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.lcd.device import st7735
from PIL.ImageDraw import ImageDraw
from PIL import ImageFont
from queue import Queue, Full

from .utils.conversation_state_types import ConversationState

# 

class FaceController(Node):

    def __init__(self):
        super().__init__('face_controller_node')


        serial_0 = spi(port=0, device=0, gpio_DC=6, gpio_RST=5)
        serial_1 = spi(port=1, device=0, gpio_DC=24, gpio_RST=23)

        self.device_0 = st7735(serial_0, rotate=1, gpio_LIGHT=26)
        self.device_1 = st7735(serial_1, rotate=1, gpio_LIGHT=25)

        # Face refresh setup
        self.face_update_callback_group = MutuallyExclusiveCallbackGroup()
        self.face_udate_timer = self.create_timer(
            0.0368, 
            self.update_face_callback,
            self.face_update_callback_group)

        # Rec result refresh
        self.rec_sub = None
        self.rec_queue = Queue(5)
        # [[self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]]

        self.conversation_state_sub = self.create_subscription(
            String,
            'conversation_state',
            self.state_callback,
            10
        )   
        self.current_state = ConversationState.IDLE
        self.last_state = ConversationState.IDLE
        
        self.temp_eye_pos_lock = threading.Lock()
        self.temp_eye_pos = [self.__lin_map(0, -1,1, 128-20, 20),self.__lin_map(0, -1,1, 35,160-35)]

    ################################################################################################
    ############## CALLBACKS ############################################
    def update_face_callback(self):
        with canvas(self.device_0) as dev0_draw, canvas(self.device_1) as dev1_draw:
            if self.current_state in ConversationState.IN_CONVERSATION:
                self.draw_tracked_square_eyes(dev0_draw, dev1_draw)
            else:
                self.draw_center_sleep_eyes(dev0_draw, dev1_draw)

            self.draw_status(self.current_state, dev0_draw, dev1_draw)
    
    def state_callback(self, msg):
        # Convert string back to ConversationState flag
        try:
            state_flag = ConversationState[msg.data]
            self.current_state = state_flag
            self.react_to_state_change(state_flag)
            self.last_state = state_flag
        except KeyError:
            self.get_logger().warn(f"Unknown state received: {msg.data}")

    def rec_frame_callback(self, msg):
        if not msg.points:
            return

        target = msg.points[0].location

        x = self.__lin_map(target.x, -1,1, 128-20, 20)
        y = self.__lin_map(target.y, -1,1, 35,160-35)

        try:
            self.rec_queue.put_nowait([x,y])
        except Full:
            pass
    
    def react_to_state_change(self, state_flag):
        # enter idle: keep still
        if state_flag == ConversationState.IDLE:
            self.destroy_subscription(self.rec_sub)

        # enter conversation: start track
        elif state_flag in ConversationState.IN_CONVERSATION and self.last_state == ConversationState.IDLE:
            self.rec_sub = self.create_subscription(
                RecFrameResult2,
                'rec_frame_result',
                self.rec_frame_callback,
                1,
                callback_group=self.face_update_callback_group
            )

        
    ################################################################################################
    ############## DRAW FUNCTIONS ############################################
    def draw_tracked_square_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw):
        if self.rec_queue.qsize() > 1:
            point = self.rec_queue.get()
            x = point[0]
            y = point[1]


            self.device_0.backlight(False)
            draw_rounded_rect_centered(dev0_draw, x, y, 40, 70, 'orange')
            
         
            self.device_1.backlight(False)
            draw_rounded_rect_centered(dev1_draw, x, y, 40, 70, 'orange')


        elif self.rec_queue.qsize() == 1:

            last_pos = self.rec_queue.get()

            with self.temp_eye_pos_lock:
                self.temp_eye_pos = last_pos

            x = self.temp_eye_pos[0]
            y = self.temp_eye_pos[1]


            self.device_0.backlight(False)
            draw_rounded_rect_centered(dev0_draw, x, y, 40, 70, fill='orange')
            
            self.device_1.backlight(False)
            draw_rounded_rect_centered(dev1_draw, x, y, 40, 70, fill='orange')
        
        else:
            x = self.temp_eye_pos[0]
            y = self.temp_eye_pos[1]

            self.device_0.backlight(False)
            draw_rounded_rect_centered(dev0_draw, x, y, 40, 70, fill=None, outline='orange')
            
            self.device_1.backlight(False)
            draw_rounded_rect_centered(dev1_draw, x, y, 40, 70, fill=None, outline='orange')
    
    def draw_status(self, state:ConversationState, dev0_draw:ImageDraw, dev1_draw:ImageDraw):
        if state in ConversationState.ROBOT_TURN:

            self.device_0.backlight(False)
            self.draw_centered_text(dev0_draw,"-- RESPONDING --", 64, 0, fill="orange")

            self.draw_centered_text(dev1_draw,"-- RESPONDING --", 64, 0, fill="orange")
        # if state == ConversationState.USER_TURN:
        #     self.device_1.backlight(False)
        #     self.draw_centered_text(dev0_draw,"** LISTENING **", 64, 0, fill="red")


    def draw_center_sleep_eyes(self, dev0_draw:ImageDraw, dev1_draw:ImageDraw):
        # x = self.__lin_map(0, -1,1, 128, 0)
        # y = self.__lin_map(0, -1,1, 0, 160)
        x = 170
        y = 170

        self.device_0.backlight(False)
        draw_rounded_rect_centered(dev0_draw, x, y, 100, 0, 'orange')
        
        self.device_1.backlight(False)
        draw_rounded_rect_centered(dev1_draw, x, y, 100, 0, 'orange')
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
    


def main(args=None):
    rclpy.init(args=args)
    node = FaceController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

