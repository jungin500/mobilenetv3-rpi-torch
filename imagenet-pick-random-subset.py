import os, sys
import shutil
from pathlib import Path
from glob import glob
import random
from tqdm import tqdm

class LabelReader(object):
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        if 'pretrained' in label_file_path:
            print("INFO: Using Pretrained label list! (not custom one)")

    def load_label(self):
        label_map = {}
        # Read label file into label map
        if os.path.isfile(self.label_file_path):
            with open(self.label_file_path, 'r') as f:
                label_name_body = f.read().strip()
                label_name_lines = label_name_body.split("\n")
                for label_entry in tqdm(label_name_lines, desc='레이블 파일 읽기 작업'):
                    synset_name, label_name = label_entry.strip().split("|")
                    label_map[synset_name] = label_name

            print(f"레이블 파일 읽기 완료: 총 {len(list(label_map.keys()))}개 레이블 검색")
            return label_map
        else:
            return None

if __name__ == '__main__':
    dataset_path = r'C:\ILSVRC2012'
    subset_path = r'C:\ILSVRC2012Subset'

    labelmap = LabelReader('imagenet_label.list').load_label()

    if os.path.isdir(subset_path):
        shutil.rmtree(subset_path, ignore_errors=True)
    os.makedirs(subset_path, exist_ok=True)

    dataset_folders = [Path(k).stem for k in glob(os.path.join(dataset_path, '*'))]

    # shuffle folders
    random.shuffle(dataset_folders)

    # select n folders from start
    source_folders = dataset_folders[:200]
    for folder_name in source_folders:
        source_path = os.path.join('..', 'ILSVRC2012', folder_name)
        target_path = os.path.join(subset_path, folder_name)

        os.symlink(source_path, target_path)
    
    # save labels
    filtered_labelmap = {}
    for folder_name in source_folders:
        filtered_labelmap[folder_name] = labelmap[folder_name]

    with open('imagenet_subset.list', 'w') as f:
        for key in filtered_labelmap.keys():
            f.write('%s|%s\n' % (key, filtered_labelmap[key]))