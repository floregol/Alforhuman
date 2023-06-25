
import pickle as pk
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio

MUSHROOM_DATAPATH = 'static/mushroom'
MUSHROOM_FILENAME = 'mushroom_data.pkl'


def download_mushroom():
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    DATASET_VERSION = 'mushroom_world_2017_16_10'
    DATASET_LINK = 'https://s3.eu-central-1.amazonaws.com/deep-shrooms/{}.zip'.format(
        DATASET_VERSION)

    with urlopen(DATASET_LINK) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall('./data')

    import pandas as pd
    import numpy as np

    DATASET_PATH = 'data/{}/'.format(DATASET_VERSION)

    mushroom_classes = pd.read_json(
        DATASET_PATH + 'mushroom_classes.json', lines=True)
    mushroom_imgs = pd.read_json(
        DATASET_PATH + 'mushroom_imgs.json', lines=True)
    mushroom_info = mushroom_imgs.merge(
        mushroom_classes, how="right", on="name_latin")

    def load_mushroom_images(folder_path, img_df):
        img_dict = {}
        for index, path in enumerate(img_df['file_path']):
            img_dict[index] = imageio.imread(folder_path + path)
        return img_dict

    img_dict = load_mushroom_images(DATASET_PATH, mushroom_info)
    i = 0
    for img in img_dict:
        if img_dict[img].shape != img_dict[0].shape:
            i = i + 1
            print(img_dict[img].shape)

    # Format the pictures to (480,480,3) by padding them with the edge values
    for img in img_dict:
        height = 480 - img_dict[img].shape[0]
        width = 480 - img_dict[img].shape[1]

        if(height % 2 == 1 & width % 2 == 1):
            height1, height2 = math.floor(height/2), math.floor(height/2) + 1
            width1, width2 = math.floor(width/2), math.floor(width/2) + 1
        elif(width % 2 == 1):
            width1, width2 = math.floor(width/2), math.floor(height/2) + 1
            height1, height2 = int(height/2), int(height/2)
        elif(height % 2 == 1):
            height1, height2 = math.floor(height/2), math.floor(height/2) + 1
            width1, width2 = int(width/2), int(width/2)
        else:
            height1, height2 = int(height/2), int(height/2)
            width1, width2 = int(width/2), int(width/2)

        if(height == 0):
            img_dict[img] = np.lib.pad(
                img_dict[img], ((0, 0), (width1, width2), (0, 0)), 'edge')
        elif (width == 0):
            img_dict[img] = np.lib.pad(
                img_dict[img], ((height1, height2), (0, 0), (0, 0)), 'edge')
        else:
            img_dict[img] = np.lib.pad(
                img_dict[img], ((height1, height2), (width1, width2), (0, 0)), 'edge')

    mushroom_info.edibility.value_counts()

    labels = mushroom_info.edibility.isin(
        ("edible", "edible and good", "edible and excellent"))

    X = []
    y = []

    for i in range(len(labels)):
        if(img_dict[i].shape == (480, 480, 3)):
            y.append(labels[i])
            X.append(img_dict[i])

    X = np.stack(X)
    y = pd.Series(y)
    data = {'X': X, 'y': y}
    return data


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

NUM_CHUNKS = 10
def storing_mushroom_dataset(data):
    X = data['X']
    y = data['y']
    indices_of_chunks = chunks(range(len(y)), int(len(y)/NUM_CHUNKS))
    for i, indices_of_chunk in enumerate(indices_of_chunks):
        smaller_X = X[indices_of_chunk, :, :, :]
        smaller_y = y[indices_of_chunk]
        print('storing smaller dict', i, '/',NUM_CHUNKS)
        chunk_data = {'X': smaller_X, 'y':smaller_y}
        file_path = os.path.join(
            MUSHROOM_DATAPATH, str(i)+'_' + MUSHROOM_FILENAME)
        with open(file_path, 'wb') as f:
            pk.dump(chunk_data, f)


def show_mushroom(x, label=None):
    plt.imshow(x)
    plt.axis('off')
    if label is None:
        plt.title('What is this ?')
    else:
        plt.title('This is '+str(label))
    plt.tight_layout()
    plt.draw()
    plt.pause(2)
    plt.close()


def draw_im(x, filename):
    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def get_mushroom_dataset():

    #data_file_path = os.path.join(MUSHROOM_DATAPATH, 'mushroom_data.pkl')

    # with open(data_file_path, "rb") as f:
    #     dataset_data = pk.load(f)

    X_list = []
    y_list = []
    for i in range(NUM_CHUNKS+1):
        data_file_path = os.path.join(
            MUSHROOM_DATAPATH, str(i)+'_' + MUSHROOM_FILENAME)
        try:
            with open(data_file_path, "rb") as f:
                chunk_dataset_data = pk.load(f)
            X_list.append(chunk_dataset_data['X'])
            y_list.append(chunk_dataset_data['y'])
        except Exception:
            pass

    
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    dataset_data = {'X': X, 'y': y}
    images_path = []
    for i, x in enumerate(X):
        image_path = os.path.join(
            MUSHROOM_DATAPATH, 'mushroom_{}.png'.format(i))
        images_path.append(image_path)
    return dataset_data['X'], np.array(dataset_data['y']), images_path


def generate_and_store_mushroom_images():
    X, y, images_path = get_mushroom_dataset()
    for i, x in enumerate(X):
        draw_im(x, images_path[i])


# Generate bunch of datasets. created in data_path, create the counter file.
if __name__ == '__main__':

    if not os.path.exists(MUSHROOM_DATAPATH):
        print('Creating the dataset folder since it wasn\'t there\n')
        os.makedirs(MUSHROOM_DATAPATH)

    try:
        print('Trying to generate mushroom images ....')

        generate_and_store_mushroom_images()

    except Exception as e:
        print('Failed.')
        print('Downloading the mushroom dataset...')
        data = download_mushroom()
        storing_mushroom_dataset(data)

        print('Attempting to generate mushroom images again....')

        generate_and_store_mushroom_images()


    