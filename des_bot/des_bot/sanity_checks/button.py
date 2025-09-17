from gpiozero import Button, LED
from signal import pause

main_button = Button(14, bounce_time=0.05)
stop_button = Button(22, bounce_time=0.05)

out1 = LED(15)
out2 = LED(27)

cnt = 1
cnt2 = 1

def on_main_button_pressed():
    global cnt
    print(f"Main button pressed: {cnt}")                 
    cnt += 1

def on_stop_button_pressed():
    global cnt2
    print(f"Stop button pressed: {cnt2}" )
    cnt2 += 1

main_button.when_pressed = on_main_button_pressed
stop_button.when_pressed = on_stop_button_pressed

print(main_button)
pause()