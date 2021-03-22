import cv2
import utils
import numpy as np
import SimpleITK as sitk

vol_raw = sitk.ReadImage("CC0001_philips_15_55_M.nii.gz")
vol = sitk.GetArrayFromImage(vol_raw)
show_vol = vol.copy()
print(vol.min(), vol.max())

s = 0
window_name = 'visual'
bar_name = 'Slice'
cv2.namedWindow(window_name)
cv2.createTrackbar(bar_name, window_name, 0, vol.shape[0] - 1, utils.nothing)  # 滚动条名，窗体名，最小值，最大值，回调函数
cv2.setMouseCallback(window_name, utils.MouseEvent)

while True:
    cv2.imshow(window_name, utils.img_normalize(show_vol, s))
    if cv2.waitKey(1) & 0xFF == ord('q'):  # quit
        break
    s = cv2.getTrackbarPos(bar_name, window_name)
    if len(utils.mouse_xy) != 0:
        show_vol = utils.mouse_xy_to_window_level(vol)

cv2.destroyAllWindows()
