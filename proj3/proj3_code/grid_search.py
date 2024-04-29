import os
import os.path as osp
import pickle
import random
import sys
from tqdm import tqdm

sys.path.append('.')
import proj3_code.student_code as sc
from proj3_code.utils import *


categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', 'Industrial', 'Suburb',
              'InsideCity', 'TallBuilding', 'Street', 'Highway', 'OpenCountry', 'Coast',
              'Mountain', 'Forest']
abbr_categories = ['Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub',
                   'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst',
                   'Mnt', 'For']
num_train_per_cat = 100
cat2idx = {cat: idx for idx, cat in enumerate(categories)}
cache_dir = 'cache'
os.makedirs(cache_dir, exist_ok=True)

print('Loading data...')
train_image_arrays, test_image_arrays, train_labels, test_labels = get_image_arrays('data',
                                                                                    categories,
                                                                                    num_train_per_cat)

def get_results(vocab_size, stride, step_size, ks):
    vocab_path = osp.join(cache_dir, 'vocab_size_{}_stride_{}.pkl'.format(vocab_size, stride))
    if not os.path.exists(vocab_path):
        vocab = sc.build_vocabulary(train_image_arrays, vocab_size, stride)
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
    else:
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)

    train_image_feats = sc.get_bags_of_sifts(train_image_arrays, vocab, step_size)
    test_image_feats = sc.get_bags_of_sifts(test_image_arrays, vocab, step_size)
    
    best_acc, best_k = 0, 0    
    for k in tqdm(ks, desc='k'):
        predicted_categories = sc.nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k=k)
        y_true = [cat2idx[cat] for cat in test_labels]
        y_pred = [cat2idx[cat] for cat in predicted_categories]
        acc = np.mean(np.array(y_true) == np.array(y_pred))
        if acc > best_acc:
            best_acc = acc
            best_k = k
    
    return best_acc, dict(vocab_size=vocab_size, stride=stride, step_size=step_size, k=best_k)


vocab_sizes = [50, 100, 200, 500]
strides = [10, 20, 30]
step_sizes = [10, 20, 30]
k = [1, 2, 5, 10, 15, 20]

for vocab_size in vocab_sizes:
    for stride in strides:
        for step_size in step_sizes:
            print('vocab_size: {}, stride: {}, step_size: {}'.format(vocab_size, stride, step_size))
            acc, params = get_results(vocab_size, stride, step_size, k)
            print('vocab_size: {}, stride: {}, step_size: {}, k: {}, acc: {}'.format(vocab_size, stride, step_size, k, acc))
            print('params: {}'.format(params))
            print('----------------------------------')
