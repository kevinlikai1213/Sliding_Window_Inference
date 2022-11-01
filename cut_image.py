import os
# import cv2
import numpy as np 
import tifffile as tiff
import json
# from osgeo import gdal, gdalconst

SAVE_PATH = './temp'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

def get_data(path_name):
    img_data =tiff.imread(path_name)
    # dataset = gdal.Open(path_name, gdalconst.GA_ReadOnly)
    # img_width = dataset.RasterXSize
    # img_height = dataset.RasterYSize
    # img_data = np.array(dataset.ReadAsArray(0, 0, img_width, img_height), dtype=float)
    return img_data.astype('uint8')

def cutim(img, cut_shape, loc):
    img = img[loc[0]:(loc[0]+cut_shape[0]), loc[1]:(loc[1]+cut_shape[1]), :]
    if img.shape!=cut_shape:
        x,y,c = img.shape
        newim = np.zeros(cut_shape).astype('uint8')
        newim[0:x, 0:y, :] = img
        img = newim
    return img

def CutWithOverlap(img_path, overlap, cut_shape, save_path=SAVE_PATH):
    def _getname(img_path, loc, orisize, overlap, cut_shape):
        basename = os.path.basename(img_path)
        basename = os.path.splitext(basename)
        name = [basename[0], str(loc[0]), str(loc[1]), str(orisize[0]), str(orisize[1]), str(overlap),str(cut_shape[0]), str(cut_shape[1])]
        return '_'.join(name)
    img = get_data(img_path)
    img_shape = img.shape
    x, y, _ = img_shape
    for xx in range(0, x, overlap):
        for yy in range(0, y, overlap):
            im = cutim(img, cut_shape, (xx, yy))
            name = _getname(img_path, (xx, yy), img_shape, overlap, cut_shape)
            tiff.imsave(os.path.join(save_path, name+'.tif'), im)

def generate_blank_coco_json(ims_path, template=None):
    def _images(imgs):
        images = []
        for i, img in enumerate(imgs):
            basename = os.path.basename(img)
            basename, _ = os.path.splitext(basename)
            info = basename.split('_')
            images.append({'file_name': img, 'height': int(info[-2]), 
                            'width': int(info[-1]), 'id': i})
        return images
    imgs = os.listdir(ims_path)
    if template:
        coco = json.load(open(template))
        coco['images'] = _images(imgs)
        coco['annotations'] = []
    else:
        coco={
            'images':_images(imgs),
            'categories':[],
            'annotations':[],
        }
    with open('cutted_coco.json', 'w') as f:
        f.write(json.dumps(coco))
    return coco


def parse_dir(dir_path, overlap, cut_shape, save_path=SAVE_PATH):
    imgs = os.listdir(dir_path)
    for img in imgs:
        img_path = os.path.join(dir_path, img)
        CutWithOverlap(img_path=img_path, overlap=overlap, cut_shape=cut_shape, save_path=save_path)


if __name__=="__main__":
    generate_blank_coco_json('./temp', r'G:\testim\instances_test.json')
    coco = json.load(open('cutted_coco.json'))
    pass
