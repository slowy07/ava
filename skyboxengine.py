from skybox_utils import *
from cv2.ximgproc import guidedFilter
import numpy as np
import synrain
import os


class SkyBox:
    def __init__(self, args):
        self.args = args
        self.load_sykbox()

        self.rainmodel = synrain.Rain(
            rain_intensity=0.0, gamma=1.0, light_correction=1.0
        )

        self.M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        self.frame_id = 0

    def tile_skybox_img(self, imgtile):
        screen_y1 = int(imgtile.shape[0] / 2 - self.args.out_size_h / 2)
        screen_x1 = int(imgtile.shape[1] / 2 - self.args.out_size_w / 2)
        imgtile = np.concatenate(
            [imgtile[screen_y1:, :, :], imgtile[0:screen_y1, :, :]], axis=0
        )
        imgtile = np.concatenate(
            [imgtile[:, screen_x1:, :], imgtile[:, 0:screen_x1, :]], axis=1
        )
        return imgtile

    def load_sykbox(self):
        print("initialize skybox ...")

        if ".jpg" in self.args.skybox:
            skybox_img = cv2.imread(
                os.path.join(r"./skybox", self.args.skybox), cv2.IMREAD_COLOR
            )
            skybox_img = cv2.cvtColor(skybox_img, cv2.COLOR_BGR2RGB)

            self.skybox_img = cv2.resize(
                skybox_img, (self.args.out_size_w, self.args.out_size_h)
            )

            cc = 1.0 / self.args.skybox_center_crop
            imgtile = cv2.resize(
                skybox_img,
                (int(cc * self.args.out_size_w), int(cc * self.args.out_size_h)),
            )
            self.skybox_imgx2 = self.tile_skybox_img(imgtile)
            self.skybox_imgx2 = np.expand_dims(self.skybox_imgx2, axis=0)

        else:
            cap = cv2.VideoCapture(os.path.join(r"./skybox", self.args.skybox))
            m_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cc = 1.0 / self.args.skybox_center_crop
            self.skybox_imgx2 = np.zeros(
                [
                    m_frames,
                    int(cc * self.args.out_size_h),
                    int(cc * self.args.out_size_w),
                    3,
                ],
                np.uint8,
            )
            for i in range(m_frames):
                _, skybox_img = cap.read()
                skybox_img = cv2.cvtColor(skybox_img, cv2.COLOR_BGR2RGB)
                imgtile = cv2.resize(
                    skybox_img,
                    (int(cc * self.args.out_size_w), int(cc * self.args.out_size_h)),
                )
                skybox_imgx2 = self.tile_skybox_img(imgtile)
                self.skybox_imgx2[i, :] = skybox_imgx2

        def skymask_refinement(self, G_pred, img):
            r, eps = 20, 0.01
            refined_skymask = guidedFilter(img[:, :, 2], G_pred[:, :, 0], r, eps)

            refined_skymask = np.stack(
                [refined_skymask, refined_skymask, refined_skymask], axis=-1
            )

            return np.clip(refined_skymask, a_min=0, a_max=1)

        def get_skybg_from_box(self, m):
            self.M = update_transformation_matrix(self.M, m)

            nbgs, bgh, bgw, c = self.skybox_imgx.shape
            fetch_id = self.frame_id % nbgs
            skybg_warp = cv2.warpAffine(
                self.skybox_imgx2[fetch_id, :, :, :],
                self.M,
                (bgw, bgh),
                borderMode=cv2.BORDER_WRAP,
            )

        def skybox_tracking(self, frame, frame_prev, skymask):
            if np.mean(skymask) < 0.05:
                print("sky area is too small")
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            prev_gray = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
            prev_gray = np.array(255 * prev_gray, dtype=np.uint8)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            curr_gray = np.array(255 * curr_gray, dtype=np.uint8)

            mask = np.array(skymask[:, :, 0] > 0.99, dtype=np.uin8)
            template_size = int(0.05 * mask.shape[0])
            mask = cv2.erode(mask, np.ones([template_size, template_size]))

            prev_pts = cv2.goodFeaturesToTrack(
                prev_gray,
                mask=mask,
                maxCorners=200,
                qualityLevel=0.01,
                minDistance=30,
                blockSize=3,
            )

            # shitomas corner detection
            if prev_pts is None:
                print("no feature point detected")
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            # calculate optical flow
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pt, None
            )

            # filter only valid points
            idx = np.where(status == 1)[0]
            if idx.size == 0:
                print("no good point matched")
                return np.array([[1, 0, 0], [0, 1, 0]])

            prev_pts, curr_pts = removeOutliners(prev_pts, curr_pts)

            if curr_pts.shape[0] < 10:
                print("no good point matched")
                return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

            dxdyda = estimate_partial_transform(
                (np.array(prev_pts), np.array(curr_pts))
            )
            m = build_transformation_matrix(dxdyda)

            return m

        def relighting(self, img, skybg, skymask):
            step = int(img.shape[0] / 20)
            skybg_thumb = skybg[::step, ::step, :]
            img_thumb = img[::step, ::step, :]
            skymask_thumb = skymask[::step, ::step, :]
            skybg_mean = np.mean(skybg_thumb, axis=(0, 1), keepdims=True)
            img_mean = np.sum(
                img_thumb * (1 - skybox_thumb), axis=(0, 1), keepdims=True
            ) / ((1 - skymask_thumb).sum(axis=(0, 1), keepdims=True) + 1e-9)
            diff = skybg_mean - img_mean
            img_colortune = img + self.args.recoloring_factor * diff

            if self.args.auto_light_matching:
                img = img_colortune
            else:
                img = self.args.relighting_factor * (
                    img_colortune + (img.mean() - img_color)
                )
            return img

        def halo(self, syneth, skybg, skymask):
            # reflection
            halo = 0.5 * cv2.blur(
                skybg * skymask,
                (int(self.args.out_size_w / 5), int(self.args.out_size_w / 5)),
            )
            syneth_with_halo = 1 - (1 - syneth) * (1 - halo)

            return syneth_with_halo

        def skyblend(self, img, img_prev, skymask):
            m = self.skybox_tracking(img, img_prev, skymask)
            skybg = self.get_skybg_from_box(m)
            img = self.relighting(img, skybg, skymask)
            syneth = img * (1 - skymask) + skybg * skymask

            if self.args.halo_effect:
                syneth = self.halo(syneth, skybg, skymask)

            if "rainy" in self.args.skybox:
                syneth = self.raindmodel.forward(syneth)

            return np.clip(syneth, a_min=0, a_max=1)
