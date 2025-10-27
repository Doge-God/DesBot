from PIL.ImageDraw import ImageDraw

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