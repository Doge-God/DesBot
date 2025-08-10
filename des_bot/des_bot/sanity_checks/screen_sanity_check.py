from luma.core.interface.serial import spi
from luma.core.render import canvas
from luma.lcd.device import st7735
from PIL.ImageDraw import ImageDraw

serial_0 = spi(port=0, device=0, gpio_DC=6, gpio_RST=5)
serial_1 = spi(port=1, device=0, gpio_DC=24, gpio_RST=23)

device_0 = st7735(serial_0, rotate=1, gpio_LIGHT=26)
device_1 = st7735(serial_1, rotate=1, gpio_LIGHT=25)

# class CenteredEllipse:
#     def __init__(self, draw:ImageDraw):

def draw_centered_ellipse(draw:ImageDraw,x,y,w,h,fill):
    draw.ellipse([(x-w/2, y-h/2), (x+w/2, y+h/2)], fill= fill)

while True:
    with canvas(device_0) as draw:
        device_0.backlight(False)
        draw_centered_ellipse(draw, 64, 80, 40, 70, 'cyan')



    
    with canvas(device_1) as draw:
        device_1.backlight(False)
        draw_centered_ellipse(draw, 64, 80, 40, 70, 'cyan')

        