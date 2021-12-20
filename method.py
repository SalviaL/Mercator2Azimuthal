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


def Mercator2Azimuthal_with_mask(img: np.ndarray,
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
            pixel = (1 - alpha) * (1 - beta) * img[int(
                np.floor(pano_h)
            ), int(np.floor(pano_w))] + alpha * (1 - beta) * img[int(
                np.ceil(pano_h)
            ), int(np.floor(pano_w))] + (1 - alpha) * beta * img[int(
                np.floor(pano_h)
            ), int(np.ceil(pano_w))] + alpha * beta * img[int(np.ceil(pano_h)),
                                                          int(np.ceil(pano_w))]
            out[v, u, :] = pixel
    kernel = np.ones((10, 10), np.uint8)
    # msk = cv2.dilate(msk, kernel, iterations=3)
    return out, msk


def Mercator2Azimuthal(mg: np.ndarray,
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
    uv = np.vstack(
        np.unravel_index(out[:, :, 0].argsort(axis=None)[::-1],
                         out[:, :, 0].shape))
    A = np.array([[1, 0], [0, -1], [0, 0]])
    B = np.array([[-out_W * 0.5], [out_H * 0.5], [f]])
    P = Rmatrix @ (A @ uv + B)
    THETA = np.arccos(P[2] / np.linalg.norm(P, 2, 0))
    FI = np.arccos(P[0] / np.linalg.norm(P[:2], 2, 0))
    FI[P[1] < 0] = 2 * PI - FI[P[1] < 0]
    U = FI * W / (2 * PI)
    V = THETA * H / PI
    U[U <= 0] = 0
    U[U >= H - 1] = H - 1
    V[V <= 0] = 0
    V[V >= W - 1] = W - 1
    UV = np.concatenate((U[:, np.newaxis], V[:, np.newaxis]), axis=1)
    UV = UV.reshape(out_W, out_H, 2).astype(np.int)

    for coor in zip(uv[0], uv[1], U, V):
        u_ = coor[0]
        v_ = coor[1]
        U_ = coor[2]
        V_ = coor[3]
        alpha = bilinear_para(V_)
        beta = bilinear_para(U_)
        pixel = (1 - alpha) * (1 - beta) * img[int(np.floor(
            V_)), int(np.floor(U_))] + alpha * (1 - beta) * img[int(np.ceil(
                V_)), int(np.floor(U_))] + (1 - alpha) * beta * img[int(
                    np.floor(V_)), int(np.ceil(U_))] + alpha * beta * img[int(
                        np.ceil(V_)), int(np.ceil(U_))]
        out[v_, u_, :] = pixel

    return out


if __name__ == "__main__":
    time0 = time.time()
    img_root = r"image/example_img.png"
    img = cv2.imread(img_root)
    plt.subplot(1, 2, 1)
    args = dict(pitch=0, heading=270, fov=124, out_H=1024, out_W=1024))
    out, msk = Mercator2Azimuthal_with_mask(img,  **args)
    plt.imshow(img[:, :, ::-1])
    plt.xlabel("original pano image")
    plt.subplot(1, 2, 2)
    plt.imshow(out[::-1, ::-1, ::-1])
    plt.xlabel(str(args)[1:-1].replace('\'', '') + str(time.time() - time0))
    plt.show()
