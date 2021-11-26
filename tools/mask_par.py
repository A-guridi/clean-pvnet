import os
import cv2
import tqdm
import numpy as np


class MaskExtract:
    def __init__(self, mask_dir, outdir, position=1):
        self.mask_dir = mask_dir
        self.all_mask = sorted(os.listdir(mask_dir))
        self.position = position
        self.output_dir = outdir + "mask/"
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def save_mask(self, idx=0):
        mask = cv2.imread(os.path.join(self.mask_dir, self.all_mask[idx]), cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask)
        mask = np.where(mask == self.position, 255, 0)
        cv2.imwrite(os.path.join(self.output_dir, self.all_mask[idx]), mask)

    def save_all(self):
        print("Saving all mask")
        for i in tqdm.tqdm(range(len(self.all_mask))):
            self.save_mask(i)


if __name__ == "__main__":
    mask_dir = "/home/arturo/datasets/test_dataset_arturo/sequence_15/mask/"
    outdir = "/home/arturo/datasets/testset/"
    position = 4
    mex = MaskExtract(mask_dir, outdir, position)
    mex.save_all()
