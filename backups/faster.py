import cv2
import numpy as np
import os
from tqdm import tqdm
import glob
from multiprocessing import Pool, Manager
from feature_extract import FeatureExtract
import gc

def extract_features(image_path):
    feat = FeatureExtract()
    feat_vec = feat.extractFeat(image_path)
    name = os.path.split(image_path)[-1].split('.')[0]
    return feat_vec, name

def process_batch(image_list):
    # Reduce memory usage by processing small batches and explicitly freeing memory
    results = []
    for image_path in image_list:
        results.append(extract_features(image_path))
    return results

def main():
    image_list = glob.glob('../VOCdevkit/VOC2012/JPEGImages/*.jpg')
    batch_size = 100  # Define a reasonable batch size
    batches = [image_list[i:i + batch_size] for i in range(0, len(image_list), batch_size)]
    
    feat_list = []
    name_list = []

    with Manager() as manager:
        pool = Pool(processes=os.cpu_count())
        manager_list = manager.list()

        for batch in batches:
            jobs = [pool.apply_async(process_batch, (batch,), callback=manager_list.extend) for batch in batches]
            # Display the progress bar
            for _ in tqdm(manager_list, total=len(image_list)):
                pass

        pool.close()
        pool.join()

        # Unpack results from jobs
        for job in jobs:
            batch_results = job.get()
            for feat_vec, name in batch_results:
                feat_list.append(feat_vec)
                name_list.append(name)

    # Convert lists to numpy arrays
    feat_list = np.array(feat_list)
    name_list = np.array(name_list)

    # Save the name list
    np.save('./VOCdevkit/name_list.npy', name_list)

    # Create and populate the FAISS index
    import faiss
    index = faiss.IndexFlatL2(512)  # Adjust the dimensionality as per your features
    index.add(feat_list)

    # Save the index to disk
    faiss.write_index(index, './VOCdevkit/voc.index')

    # Clear memory
    del feat_list, name_list
    gc.collect()

if __name__ == '__main__':
    main()
