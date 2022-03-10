import glob
import json
import os
import yaml
import matplotlib.pyplot as plt
import random
import matplotlib.image as mpimg


def get_config(yaml_file='./config.yml'):
    with open(yaml_file, 'r') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)
    return cfgs


def get_data(data_path, show = True, demo_img_num = 4, col_num = 2):
    images = glob.glob(os.path.join(data_path,"*/*.*"))
    if show:
        print("There are {0} images in {1} folder".format(len(images), data_path.split('/')[-2]))
        print("This is a demo of {} images".format(demo_img_num))

        rows = demo_img_num//col_num
        for i in range(1, demo_img_num+1):
            image = random.choice(images)
            image = mpimg.imread(image)
            plt.subplot(rows,col_num,i)
            plt.imshow(image)
        plt.show()
    return images


def read_json(file):
    with open(file,'r') as f:
        data = json.load(f)
    return data


def write_json(file,data):
    with open(file,'w') as f:
        json.dump(data,f,indent = 4)


def make_data(cfgs):
    train_images = get_data(os.path.join(cfgs['data']['ROOT'], cfgs['data']['TRAIN_FOLDER']), show = False)
    val_images = get_data(os.path.join(cfgs['data']['ROOT'],cfgs['data']['VAL_FOLDER']), show = True)

    random.shuffle(train_images)
    random.shuffle(val_images)

    write_json(cfgs['data']['train_path'], train_images)
    write_json(cfgs['data']['val_path'], val_images)


def get_label(path):
    lookup = {str(k): k for k in range(1,10)}
    try:
        return lookup[path.split('/')[-2]] - 1
    except:
        return None


if __name__=='__main__':
    cfgs = get_config()
    make_data(cfgs)

