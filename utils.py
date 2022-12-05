import os
# import cv2
import numpy as np 
import tifffile as tiff
import json
import cv2
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
    img_data = img_data/np.max(img_data)*255
    return img_data.astype('uint8')

def cutim(img, cut_shape, loc):
    img = img[loc[0]:(loc[0]+cut_shape[0]), loc[1]:(loc[1]+cut_shape[1]), :]
    if img.shape!=cut_shape:
        x,y,c = img.shape
        newim = np.zeros(cut_shape).astype('uint8')
        print(newim.shape)
        newim[0:x, 0:y, :] = img
        img = newim
    return img

def CutWithOverlap(img_path, overlap, cut_shape, save_path=SAVE_PATH):
    def _getname(img_path, loc, orisize, overlap, cut_shape):
        basename = os.path.basename(img_path)
        basename = os.path.splitext(basename)
        name = [basename[0], str(loc[0]), str(loc[1]), str(orisize[0]), str(orisize[1]), str(overlap),str(cut_shape[0]), str(cut_shape[1])]
        return '_'.join(name)
    img= get_data(img_path)
    img = img[:, :, :3]
    loc =  (img[:,:,0]==255) &( img[:,:,1]==255 )& (img[:,:,2]==255) 
    img[loc, :] = [0,0,0]
    img_shape = img.shape
    x, y, _ = img_shape
    stepx = cut_shape[0]-overlap
    stepy = cut_shape[1]-overlap
    # tiff.imsave(os.path.join('0.tif'),  img)
    for xx in range(0, x, stepx):
        if (xx+cut_shape[0])>x:
            xx = x-cut_shape[0]
        for yy in range(0, y, stepy):
            flagy=False
            

            if (yy+cut_shape[1])>y:
                yy = y-cut_shape[1]
            im = cutim(img, cut_shape, (xx, yy))
            name = _getname(img_path, (xx, yy), img_shape, overlap, cut_shape)
            tiff.imsave(os.path.join(save_path, name+'.tif'), im )
            # cv2.imwrite(os.path.join(save_path, name+'.tif'), im)
            print('save : {} \r'.format(os.path.join(save_path, name+'.tif')))


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

def merge_dir(dir_path, SAVE_MERGE_PATH='./'):
    def parse_name(path):
        basename = os.path.basename(path)
        name = basename.split('.')[0]
        return name.split('_')
    imgs = os.listdir(dir_path)
    info = parse_name(imgs[0])
    img_matrix = cv2.imread(os.path.join(dir_path, imgs[0]))
    try:
        w, h, c = img_matrix.shape
    except:
        c=0
    name, x, y, orx, ory, ovrlp, ctshpx, ctshpy = info[0], int(info[1]), int(info[2]), int(info[3]), int(info[4]), int(info[5]), int(info[6]), int(info[7])
    if c:
        img_large = np.zeros((orx,ory,c))
        tmplt_large = np.zeros((orx,ory,c))
        tmplate = np.ones((ctshpx,ctshpy,c))
    else:
        img_large = np.zeros((orx,ory))
        tmplt_large = np.zeros((orx,ory))
        tmplate = np.ones((ctshpx,ctshpy))
        
    for img in imgs:
        img_path = os.path.join(dir_path, img)
        img_matrix = cv2.imread(img_path)
        info = parse_name(img_path)
        name, x, y, orx, ory= info[0], int(info[1]), int(info[2]), int(info[3]), int(info[4])
        img_large[x:x+ctshpx, y:y+ctshpy,:] += img_matrix
        tmplt_large[x:x+ctshpx, y:y+ctshpy,:]  += tmplate
    img = img_large/tmplt_large
    img = img.astype('uint8')
    tiff.imsave(os.path.join(SAVE_MERGE_PATH, name+'.tif'), img)
    
if __name__=="__main__":
    parse_dir(r'E:\WeChatFiles\WeChat Files\wxid_7umnurdlztzj22\FileStorage\File\2022-11\测试数据\测试数据\32648\3005\97\2020', 128, (512,512,3))
    merge_dir('./temp')
