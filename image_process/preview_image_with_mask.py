import argparse
import json 
from PIL import Image, ImageDraw
from IPython.display import display
from shapely import wkt
from os import path, makedirs, listdir

def show_polygons(img_path, json_path):
    with open(json_path, 'r', encoding='utf-8') as image_json_file:
        image_json = json.load(image_json_file)
    coords = image_json['features']['xy']

    polygons = []

    if(len(coords) != 0):
        for coord in coords:
            if 'subtype' in coord['properties']:
                damage = coord['properties']['subtype']
            else:
                damage = 'no-damage'
            polygons.append((damage, wkt.loads(coord['wkt'])))


    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, 'RGBA')

    damage_dict = {
    "no-damage": (0, 255, 0, 125),
    "minor-damage": (255, 255, 0, 125),
    "major-damage": (255, 128, 0, 125),
    "destroyed": (255, 0, 0, 125),
    "un-classified": (0, 255, 0, 125)
    }

    for damage, polygon in polygons:
        x,y = polygon.exterior.coords.xy
        coords = list(zip(x,y))
        draw.polygon(coords, damage_dict[damage])

    del draw
    return img

if __name__=="__main__":

    parser = argparse.ArgumentParser("PyTorch Xview Pipeline")
    arg = parser.add_argument
    arg('--image_path', type=str, default='', help='Image path')
    arg('--json_path', type=str, default='', help='Polygon path')

    args = parser.parse_args()

    if args.image_path and args.json_path:
        img_path = args.image_path
        json_path = args.json_path
    else:
        train_dirs = ['data/train', 'data/test']
        all_files = []
        for train in train_dirs:
            for f in listdir(path.join(train, 'images')):
                all_files.append(path.join(train, 'images', f))

        img_path = all_files[1]
        json_path = img_path.replace('/images/', '/labels/').replace('.png', '.json')

    print(img_path, "\n", json_path)
    
    try:
        mixed = show_polygons(img_path, json_path)
        mixed.show()
    except Exception as e:
        print(e)
