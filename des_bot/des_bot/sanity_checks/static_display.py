from gpiozero import Button, LED, PWMLED
from signal import pause

main_button = Button(14, bounce_time=0.05)
stop_button = Button(22, bounce_time=0.05)

out1 = LED(15)
out2 = LED(27)


antenna_led = PWMLED(2)
is_ant_on = False
main_button_led = PWMLED(3)
is_main_on = False

cnt = 1
cnt2 = 1

def on_main_button_pressed():
    global cnt
    global is_main_on
    print(f"Main button pressed: {cnt}") 
    if not is_main_on:
        main_button_led.pulse(fade_in_time=0.3, fade_out_time=0.5)     
        is_main_on = True
    else:
        main_button_led.off()
        is_main_on = False
    cnt += 1

def on_stop_button_pressed():
    global cnt2
    global is_ant_on
    print(f"Stop button pressed: {cnt2}" )
    if not is_ant_on:
        antenna_led.pulse(fade_in_time=0.1, fade_out_time=0.3)  
        is_ant_on = True
    else:
        antenna_led.off()
        is_ant_on = False
    cnt2 += 1

main_button.when_pressed = on_main_button_pressed
stop_button.when_pressed = on_stop_button_pressed

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

def draw_centered_ellipse_top_half(draw: ImageDraw, x, y, w, h, fill, shown_ratio=0.6):
    """
    Draws a centered ellipse, showing only the top portion as defined by shown_ratio (0-1).
    The rest is blocked out with black.
    """
    # Draw full ellipse
    draw.ellipse([(x - w/2, y - h/2), (x + w/2, y + h/2)], fill=fill)
    # Calculate the y coordinate where to start blocking
    block_start_y = y - h/2 + h * shown_ratio
    # Draw black rectangle over the bottom part
    draw.rectangle([(x - w/2, block_start_y), (x + w/2, y + h/2)], fill="black")

def draw_centered_rectangle(draw: ImageDraw, x, y, w, h, fill, radius=None):
    """
    Draws a centered rounded rectangle. If Pillow supports rounded_rectangle it will be used,
    otherwise a manual pieslice/rectangle approach is used as a fallback.
    """
    left = x - w / 2
    top = y - h / 2
    right = x + w / 2
    bottom = y + h / 2

    # determine corner radius
    if radius is None:
        r = int(min(w, h) * 0.2)
    else:
        r = int(radius)
    # clamp radius
    r = max(0, min(r, int(min(w, h) / 2)))

    draw.rounded_rectangle([(left, top), (right, bottom)], radius=r, fill=fill)


# draw_centered_ellipse_top_half(draw, 64, 80, 70, 70, '#ffc300', 0.75)
# draw_centered_rectangle(draw, 64, 80, 70, 40, '#ffc300', 10)


while True:
    with canvas(device_0) as draw:
        device_0.backlight(False)
        draw_centered_ellipse_top_half(draw, 64, 80, 70, 70, '#ffc300', 1)

    with canvas(device_1) as draw:
        device_1.backlight(False)
        draw_centered_rectangle(draw, 64, 80, 70, 30, '#ffc300', 15)