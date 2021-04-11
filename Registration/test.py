import os
import cv2
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

root_path = "E:\\Code\\my_MedImageComputing\\Registration\\data\\training_001"

# 图像读取
ct_raw = sitk.ReadImage(os.path.join(root_path, "ct", "training_001_ct.mhd"))
print(ct_raw.GetSize())
print(ct_raw.GetSpacing())

mrt1_raw = sitk.ReadImage(os.path.join(root_path, "mr_T1", "training_001_mr_T1.mhd"))
print(mrt1_raw.GetSize())
print(mrt1_raw.GetSpacing())

# 图像重采样
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(ct_raw)
resampler.SetSize(ct_raw.GetSize())
resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
mrt1_raw_resample = resampler.Execute(mrt1_raw)

print("CT: Size: {}, Spacing: {}".format(ct_raw.GetSize(), ct_raw.GetSpacing()))
print("MR_T1: Size: {}, Spacing: {}".format(mrt1_raw_resample.GetSize(), mrt1_raw_resample.GetSpacing()))

ct_array = sitk.GetArrayFromImage(ct_raw)
mrt1_array = sitk.GetArrayFromImage(mrt1_raw_resample)

print(ct_array.shape)
print(mrt1_array.shape)

# 图像可视化与点标记
ct_point = []
mrt1_point = []

fig, axes = plt.subplots(1, 2)
axes[0].imshow(ct_array[0], cmap='gray')
axes[1].imshow(mrt1_array[0], cmap='gray')

ct_axes = plt.axes([0.2, 0.10, 0.65, 0.03])
mrt1_axes = plt.axes([0.2, 0.05, 0.65, 0.03])
ct_slice = Slider(ct_axes, "CT", valmin=0, valmax=ct_array.shape[0] - 1, valfmt="%.0f", valinit=0, valstep=1)
mrt1_slice = Slider(mrt1_axes, "MR_T1", valmin=0, valmax=mrt1_array.shape[0] - 1, valfmt="%.0f", valinit=0, valstep=1)


# 定义滑条
def update_ct(val):
    global ct_point
    s = ct_slice.val
    axes[0].clear()
    axes[0].set(title="CT")
    axes[0].imshow(ct_array[s], cmap='gray')
    axes[0].axis("off")
    for point in ct_point:
        z, x, y = point
        if s == z:
            axes[0].scatter(x, y, marker="+", c='lime')


def update_mrt1(val):
    global mrt1_point
    s = mrt1_slice.val
    axes[1].clear()
    axes[1].set(title="MR_T1")
    axes[1].imshow(mrt1_array[s], cmap='gray')
    axes[1].axis("off")
    for point in mrt1_point:
        z, x, y = point
        if s == z:
            axes[1].scatter(x, y, marker="+", c='yellow')


ct_slice.on_changed(update_ct)
ct_slice.reset()
ct_slice.set_val(0)

mrt1_slice.on_changed(update_mrt1)
mrt1_slice.reset()
mrt1_slice.set_val(0)


# 定义鼠标交互
def on_press(event):
    global ct_point, mrt1_point
    x_coord, y_coord = event.xdata, event.ydata
    if event.inaxes == axes[0]:
        z_coord = ct_slice.val
        ct_point.append([z_coord, x_coord, y_coord])
        axes[0].scatter(x_coord, y_coord, marker="+", c='lime')
        axes[0].figure.canvas.draw()
    elif event.inaxes == axes[1]:
        z_coord = mrt1_slice.val
        mrt1_point.append([z_coord, x_coord, y_coord])
        axes[1].scatter(x_coord, y_coord, marker="+", c='yellow')
        axes[1].figure.canvas.draw()


fig.canvas.mpl_connect('button_press_event', on_press)
# plt.show()

# JUST FOR TESTING!
ct_point_array = \
    np.array([[5., 187.99032258, 126.58094624],
              [5., 243.25053763, 102.35729032],
              [5., 301.53870968, 128.85191398],
              [5., 242.49354839, 250.7271828],
              [5., 237.9516129, 366.54653763],
              [5., 98.6655914, 309.01535484],
              [5., 312.13655914, 291.60460215],
              [5., 386.32150538, 308.25836559]])
mrt1_point_array = \
    np.array([[0., 204.91935484, 77.37664516],
              [0., 259.42258065, 50.12503226],
              [0., 319.22473118, 75.10567742],
              [0., 269.26344086, 208.33578495],
              [0., 274.56236559, 324.91212903],
              [0., 133.76236559, 267.38094624],
              [0., 341.9344086, 242.40030108],
              [0., 414.60537634, 256.02610753]])

# SVD配准
# 数据预处理，设定旋转中心
# print("Same amount({}) of points!".format(len(ct_point)) if len(ct_point) == len(mrt1_point)
#       else "Unequal amount of Points!")
# ct_point_array, mrt1_point_array = np.array(ct_point), np.array(mrt1_point)
ct_point_centered = ct_point_array.copy()
mrt1_point_centered = mrt1_point_array.copy()
ct_point_centered[:, 0] -= ct_point_centered.mean(axis=0)[0]
mrt1_point_centered[:, 0] -= mrt1_point_centered.mean(axis=0)[0]
center = np.array(ct_array.shape[1:]) / 2
ct_point_centered[:, 1:] -= center
mrt1_point_centered[:, 1:] -= center

# 构造优化矩阵，SVD求解得到旋转矩阵和平移向量
W = ct_point_centered.T.dot(mrt1_point_centered)
U, Sigma, V_transpose = np.linalg.svd(W)
R = U.dot(V_transpose)
t = ct_point_array.mean(axis=0) - mrt1_point_array.mean(axis=0).dot(R)
new_mrt1_point_array = mrt1_point_array.dot(R) + t
print(R, t)

# 计算旋转误差 L2范数
error = np.linalg.norm(ct_point_array - new_mrt1_point_array)
print("error is %.5f" % error)

# 把2d的MR图像旋转映射到2d的CT上，用opencv所以要把图像处理一下
h, w = ct_array[5].shape
R_2d = np.zeros((2, 3))
R_2d[:, :2] = R[1:, 1:]
R_2d[:, 2] = t[1:]
old_mrt1_slice = 255 * ((mrt1_array[0] - mrt1_array[0].min()) / (mrt1_array[0].max() - mrt1_array[0].min()))
new_mrt1_slice = cv2.warpAffine(np.uint8(old_mrt1_slice), R_2d, dsize=(w, h), flags=cv2.INTER_LINEAR)

# 分开展示图片和配准后的点
fig, axes = plt.subplots(1, 3)
axes[0].imshow(ct_array[5], cmap='gray')
axes[0].scatter(ct_point_array[:, 1], ct_point_array[:, 2], marker="+", c='red', label='CT')
axes[0].scatter(mrt1_point_array[:, 1], mrt1_point_array[:, 2], marker="+", c='cyan', label='old_MR')
axes[0].scatter(new_mrt1_point_array[:, 1], new_mrt1_point_array[:, 2], marker="+", c='blue', label='new_MR')
axes[0].legend(loc='best')
axes[0].axis('off')
axes[1].imshow(mrt1_array[0], cmap='gray')
axes[1].scatter(mrt1_point_array[:, 1], mrt1_point_array[:, 2], marker="+", c='cyan', label='old_MR')
axes[1].legend(loc='best')
axes[1].axis('off')
axes[2].imshow(new_mrt1_slice, cmap='gray')
axes[2].scatter(new_mrt1_point_array[:, 1], new_mrt1_point_array[:, 2], marker="+", c='blue', label='new_MR')
axes[2].legend(loc='best')
axes[2].axis('off')
plt.show()

# 叠加展示
ct_slice_5 = 255 * ((ct_array[5] - ct_array[5].min()) / (ct_array[5].max() - ct_array[5].min()))
fig, ax = plt.subplots(1)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)
ax.imshow(ct_slice_5.clip(100, 255), cmap='gray')
ax.scatter(new_mrt1_point_array[:, 1], new_mrt1_point_array[:, 2], marker="+", c='blue', label='new_MR')
ax.imshow(new_mrt1_slice, cmap='gray', alpha=0.5)
ax.axis('off')

window_axes = plt.axes([0.2, 0.10, 0.65, 0.03])
alpha_axes = plt.axes([0.2, 0.05, 0.65, 0.03])
window_slider = Slider(window_axes, "window", valmin=0, valmax=254, valfmt="%.0f", valinit=0, valstep=1)
alpha_slider = Slider(alpha_axes, "opacity", valmin=0.0, valmax=1.0, valfmt="%.2f", valinit=0.0, valstep=0.01)


def update_window(val):
    window = window_slider.val
    ax.imshow(ct_slice_5.clip(window, 255), cmap='gray')


def update_alpha(val):
    a = alpha_slider.val
    ax.imshow(new_mrt1_slice, cmap='gray', alpha=a)


window_slider.on_changed(update_window)
window_slider.reset()
window_slider.set_val(0)

alpha_slider.on_changed(update_alpha)
alpha_slider.reset()
alpha_slider.set_val(0.0)
plt.show()
