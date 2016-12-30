import json
import os

# person 1
# car 3
class_name = {}
class_name[1]='person'
class_name['1']='person'
class_name['3']='car'
class_name[3]='car'

def parse(folder, file_name, xml_folder):
    file_name = os.path.join(folder, file_name)
    print 'parsing',file_name
    with open(file_name) as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
        new_categorie_ids = []
        for item in categories:
            if item['name']=='car' or item['name']=='person':
                if int(item['id']) not in new_categorie_ids:
                    new_categorie_ids.append(int(item['id']))
        del categories
        img_dict = {}
        for item in images:
            if 'id' in item and 'width' in item and 'height' in item and 'file_name' in item:
                img_dict[item['id']] = []
                img_dict[item['id']].append(item['file_name'])
                img_dict[item['id']].append(item['width'])
                img_dict[item['id']].append(item['height'])
        del images
        for item in annotations:
            if int(item['category_id']) in new_categorie_ids:
                image_id = item['image_id']
                bbox = item['bbox']
                bbox.append(item['category_id'])
                img_dict[image_id].append(bbox)
        del annotations
        for item in img_dict:
            if len(img_dict[item])>3:
                obj_list = []
                for box in img_dict[item][3:]:
                    obj_list.append('<object><name>%s</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>%d</xmin><ymin>%d</ymin><xmax>%d</xmax><ymax>%d</ymax></bndbox></object>'%(class_name[box[-1]], box[0], box[1], box[0]+box[2], box[1]+box[3]))
                xml_data = '<annotation><folder>COCO</folder><filename>%s</filename><source><database>Intel Database</database><annotation>Intel</annotation><image>CP</image></source><segmented>0</segmented><size><width>%s</width><height>%s</height><depth>3</depth></size>%s</annotation>'%(img_dict[item][0], img_dict[item][1], img_dict[item][2], ''.join(obj_list))
                with open(os.path.join(xml_folder, img_dict[item][0].split('.')[0]+'.xml'), 'w') as fout:
                    fout.write(xml_data)

if __name__=='__main__':
    folder = 'annotations-2014-instances'
    xml_folder = 'COCOAnn'
    files = os.listdir(folder)
    for f in files:
        parse(folder, f, xml_folder)
