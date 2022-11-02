import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

PI = np.pi


def bilinear_para(a: float):
    top_a = np.ceil(a)
    btm_a = np.floor(a)
    if abs(top_a - btm_a) < 1e-5:
        return 1
    return (a - btm_a) / (top_a - btm_a)


def Mercator2Azimuthal_detail(img: np.ndarray,
                                 out_W: int = 0,
                                 out_H: int = 0,
                                 heading: float = 0,
                                 pitch: float = 90,
                                 fov: float = 120) -> np.ndarray:
    pitch = (pitch - 90) * PI / 180
    heading = heading * PI / 180

    W = img.shape[1]
    H = img.shape[0]
    FOV = fov * PI / 180
    R = W / PI / 2
    if (out_H == 0) or (out_W == 0):
        f = R * np.sqrt(
            (1 + 0.5 * np.sin(0.5 * FOV)) * (1 - 0.5 * np.sin(0.5 * FOV)))
        out_H = int(np.round(2 * f * np.tan(0.5 * FOV)))
        out_W = out_H
    else:
        f = (0.5 * out_W) / np.tan(FOV * 0.5)
    Rz = np.array([[np.cos(heading), -np.sin(heading), 0],
                   [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Rmatrix = Rz @ Rx
    out = np.zeros((out_H, out_W, 3), np.uint8)
    msk = np.zeros((H, W), dtype=np.uint8)
    for u in range(out_W):
        for v in range(out_H):
            x = u - out_W * 0.5
            y = out_H * 0.5 - v
            z = f
            new_coor = Rmatrix @ np.transpose(np.array([[x, y, z]]))
            x = new_coor[0][0]
            y = new_coor[1][0]
            z = new_coor[2][0]
            theta = np.arccos(z / np.sqrt(x * x + y * y + z * z))
            fi = np.arccos(x / np.sqrt(x * x + y * y))
            if (y < 0):
                fi = 2 * PI - fi
            U = fi * W / (2 * PI)
            V = theta * H / PI
            pano_h = max(0, min(V, H - 1))
            pano_w = max(0, min(U, W - 1))
            msk[int(pano_h), int(pano_w)] = 255
            alpha = bilinear_para(pano_h)
            beta = bilinear_para(pano_w)
            pixel = (1 - alpha) * (1 - beta) * img[int(np.floor(
                pano_h)), int(np.floor(pano_w))] + alpha * (1 - beta) * img[
                    int(np.ceil(pano_h)),
                    int(np.floor(pano_w))] + (1 - alpha) * beta * img[
                        int(np.floor(pano_h)),
                        int(np.ceil(pano_w))] + alpha * beta * img[
                            int(np.ceil(pano_h)),
                            int(np.ceil(pano_w))]
            out[v, u, :] = pixel
    kernel = np.ones((10, 10), np.uint8)
    msk = cv2.dilate(msk, kernel, iterations=3)
    return out, msk


def Mercator2Azimuthal(img: np.ndarray,
                                 out_W: int = 0,
                                 out_H: int = 0,
                                 heading: float = 0,
                                 pitch: float = 90,
                                 fov: float = 120) -> np.ndarray:
    pitch = (pitch - 90) * PI / 180
    heading = heading * PI / 180

    W = img.shape[1]
    H = img.shape[0]
    FOV = fov * PI / 180
    R = W / PI / 2
    if (out_H == 0) or (out_W == 0):
        f = R * np.sqrt(
            (1 + 0.5 * np.sin(0.5 * FOV)) * (1 - 0.5 * np.sin(0.5 * FOV)))
        out_H = int(np.round(2 * f * np.tan(0.5 * FOV)))
        out_W = out_H
    else:
        f = (0.5 * out_W) / np.tan(FOV * 0.5)
    Rz = np.array([[np.cos(heading), -np.sin(heading), 0],
                   [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    Rmatrix = Rz @ Rx
    out = np.zeros((out_H, out_W, 3), np.uint8)
    msk = np.zeros((H, W), dtype=np.uint8)
    u_list = np.resize(np.arange(0, out_W, 1), (out_W, out_H)).flatten() 
    v_list = np.resize(np.arange(0, out_H, 1), (out_W, out_H)).T.flatten()
    z_list = np.ones(out_H * out_W) * f
    image_coor = np.vstack([u_list- out_W * 0.5, out_H * 0.5 - v_list, z_list])
    xyz_coor = Rmatrix @ image_coor
    x, y, z = xyz_coor[0], xyz_coor[1], xyz_coor[2]
    Theta = np.arccos(z / np.linalg.norm(xyz_coor, axis=0))
    Fi = np.arccos(x / np.linalg.norm(xyz_coor[:2], axis=0))
    Fi[y < 0] = 2 * PI - Fi[y < 0]
    U = Fi*W/(2*PI)
    V = Theta*H/PI
    Pano_h = np.where(V<H-1,V,H-1)
    Pano_h = np.where(Pano_h>0,Pano_h,0)
    Pano_w = np.where(U<W-1,U,W-1)
    Pano_w = np.where(Pano_w,Pano_w,0)

    msk[Pano_h.astype(np.int16),Pano_w.astype(np.int16)] = 255
    out[v_list,u_list] = img[Pano_h.astype(np.int16),Pano_w.astype(np.int16)]
    kernel = np.ones((10, 10), np.uint8)
    msk = cv2.dilate(msk, kernel, iterations=2)
    return out, msk


if __name__ == "__main__":
    time0 = time.time()
    img_root = r"image/example_img.png"
    img = cv2.imread(img_root)
    plt.subplot(1, 2, 1)
    args = dict(pitch=15, heading=110, fov=124, out_H=1024, out_W=1024)
    out, msk = Mercator2Azimuthal(img, **args)
    img[msk==255,2] = 255
    plt.imshow(img[:, :, ::-1])
    plt.xlabel("original pano image")
    plt.subplot(1, 2, 2)
    plt.imshow(out[::-1, ::-1, ::-1])
    plt.xlabel(str(args)[1:-1].replace('\'', '') + str(time.time() - time0))
    plt.show()
