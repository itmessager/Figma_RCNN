def cam2screen(x_cam, y_cam, device_params, orientation=1):
    (w_screen_cm, h_screen_cm, w_screen_pixels, h_screen_pixels, dx_cm, dy_cm) = device_params

    if orientation == 1:
        x_screen_cm = x_cam + dx_cm
        y_screen_cm = -y_cam - dy_cm

        x_screen_pixels = int(round(x_screen_cm * w_screen_pixels / w_screen_cm))
        y_screen_pixels = int(round(y_screen_cm * h_screen_pixels / h_screen_cm))
    else:
        raise Exception("Orientation not supported")

    return x_screen_pixels, y_screen_pixels