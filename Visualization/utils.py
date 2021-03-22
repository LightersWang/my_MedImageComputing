import cv2
import numpy as np

mouse_flag = False
mouse_xy = []
last_pos = None
last_width = None


def nothing(x):
    pass


def img_normalize(vol, slice):
    vol = 255 * (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
    return vol[slice].astype('uint8')


def window_level_change(vol, pos, width):
    pos = pos if pos > 0 else 0
    width = width if width > 0 else 0
    vol = vol.clip(pos - width / 2, pos + width / 2)
    return vol, pos, width


def mouse_xy_to_window_level(vol):
    global mouse_xy, last_pos, last_width

    # mouse
    x_delta = mouse_xy[-1][0] - mouse_xy[0][0]
    y_delta = mouse_xy[-1][1] - mouse_xy[0][1]
    print(f"y_delta: {y_delta}   x_delta: {x_delta}")
    width_mode = True if np.abs(x_delta) > np.abs(y_delta) else False  # True for width mode, False for pos mode

    # vol
    if last_pos is None and last_width is None:
        last_pos = (vol.max() - vol.min()) / 2
        last_width = vol.max() - vol.min()
    if width_mode is True:
        new_width = last_width + x_delta * 10
        new_pos = last_pos
    else:
        new_width = last_width
        new_pos = last_pos - y_delta * 10
    print(f"new_pos: {new_pos}   new_width: {new_width}")
    vol, last_pos, last_width = window_level_change(vol, new_pos, new_width)
    return vol


def MouseEvent(event, x, y, flags, param):
    global mouse_flag, mouse_xy
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_flag = True
    elif flags == cv2.EVENT_FLAG_LBUTTON and mouse_flag is True:
        mouse_xy.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_flag = False
        mouse_xy = []
