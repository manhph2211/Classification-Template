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
    images = glob.glob(os.path.join(data_path,"*.jpg"))
    if show:
        print("There are {0} images in {1} folder".format(len(images), "/".join(data_path.split('/')[-2:])))
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
    cat_images = get_data(cfgs['data']['TRAIN_CAT_FOLDER_PATH'], show = False)
    dog_images = get_data(cfgs['data']['TRAIN_DOG_FOLDER_PATH'], show = False)
    test_images = get_data(cfgs['data']['TEST_FOLDER_PATH'], show = False)

    val_images = []
    train_images = []
    train_val_images = cat_images + dog_images
    random.shuffle(train_val_images)
    for i, image in enumerate(train_val_images):
        if i+1 <= len(train_val_images) * cfgs['data']['train_val_size']:
            val_images.append(image)
        else:
            train_images.append(image)
    write_json(cfgs['data']['train_path'], train_images)
    write_json(cfgs['data']['val_path'], val_images)
    write_json(cfgs['data']['test_path'], test_images)


def get_label(name):
    lookup = {'dogs': 1, 'cats': 0}
    try:
        return lookup[name.split('/')[-2]]
    except:
        return name.split('/')[-1]


if __name__=='__main__':
    cfgs = get_config()
    get_data(cfgs['data']['TRAIN_CAT_FOLDER_PATH'], False, cfgs['data']['demo_img_num'])
    get_data(cfgs['data']['TRAIN_DOG_FOLDER_PATH'], False, cfgs['data']['demo_img_num'])
    get_data(cfgs['data']['TEST_FOLDER_PATH'], False, cfgs['data']['demo_img_num'])
    make_data(cfgs)

