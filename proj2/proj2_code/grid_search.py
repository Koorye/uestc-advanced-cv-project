import os
import pandas as pd
import sys
import torch
import torchvision.transforms as transforms
from plotnine import *
from tqdm import tqdm
sys.path.append("../")

from proj2_code.HarrisNet import get_interest_points
from proj2_code.SIFTNet import get_siftnet_features
from proj2_code.utils import load_image, PIL_resize, rgb2gray
from proj2_code.utils import evaluate_correspondence
from student_feature_matching import match_features


# Notre Dame
image1 = load_image('../data/1a_notredame.jpg')
image2 = load_image('../data/1b_notredame.jpg')
eval_file = '../ground_truth/notredame.pkl'

scale_factor = 0.5
image1 = PIL_resize(image1, (int(image1.shape[1]*scale_factor), int(image1.shape[0]*scale_factor)))
image2 = PIL_resize(image2, (int(image2.shape[1]*scale_factor), int(image2.shape[0]*scale_factor)))

image1_bw = rgb2gray(image1)
image2_bw = rgb2gray(image2)

#convert images to tensor
tensor_type = torch.FloatTensor
torch.set_default_tensor_type(tensor_type)
to_tensor = transforms.ToTensor()

image_input1 = to_tensor(image1_bw).unsqueeze(0)
image_input2 = to_tensor(image2_bw).unsqueeze(0)

x1, y1, _ = get_interest_points(image_input1)
x2, y2, _ = get_interest_points(image_input2)
x1, x2 = x1.detach().numpy(), x2.detach().numpy()
y1, y2 = y1.detach().numpy(), y2.detach().numpy()

image1_features = get_siftnet_features(image_input1, x1, y1)
image2_features = get_siftnet_features(image_input2, x2, y2)

mode = 'euclidean'
dist_alpha = 0.1
cosine_alpha = 0.7
ratio_thresh = 0.8
spatial_cosine_thresh = 0.6
spatial_dist_thresh = 1.0

dist_modes = ['euclidean', 'manhattan', 'chebyshev', 'minowski']
dist_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
cosine_alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ratio_threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
spatial_cosine_threshs = [-1.0, -0.8, 0.6, 0.4, 0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
spatial_dist_threshs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]

os.makedirs('images', exist_ok=True)

# search mode
accs = []
for mode_ in tqdm(dist_modes):
    matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
                                          mode=mode_, dist_alpha=dist_alpha, cosine_alpha=cosine_alpha,
                                          ratio_thresh=ratio_thresh, spatial_cosine_thresh=spatial_cosine_thresh, 
                                          spatial_dist_thresh=spatial_dist_thresh)
    try:
        num_pts_to_evaluate = 100
        acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
                                        x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
                                        x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
    except:
        acc = 0.0
    accs.append(acc)
    
df = pd.DataFrame({'mode': dist_modes, 'acc': accs})

(
    ggplot(df, aes(x='mode', y='acc', fill='mode')) +
    geom_col(show_legend=False) +
    labs(title='Accuracy vs Distance Mode', x='Distance Mode', y='Accuracy') +
    theme_seaborn()
).save('images/accuracy_vs_distance_mode.png')

# # search dist_alpha
# accs = []
# for dist_alpha_ in tqdm(dist_alphas):
#     matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
#                                           mode=mode, dist_alpha=dist_alpha_, cosine_alpha=cosine_alpha,
#                                           ratio_thresh=ratio_thresh, spatial_cosine_thresh=spatial_cosine_thresh, 
#                                           spatial_dist_thresh=spatial_dist_thresh)
#     num_pts_to_evaluate = 100
#     acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
#                                      x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
#                                      x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
#     accs.append(acc)
    
# df = pd.DataFrame({'dist_alpha': dist_alphas, 'acc': accs})

# (
#     ggplot(df, aes(x='dist_alpha', y='acc')) +
#     geom_point() +
#     geom_line() +
#     labs(title='Accuracy vs Distance Alpha', x='Distance Alpha', y='Accuracy') +
#     theme_seaborn()
# ).save('images/accuracy_vs_distance_alpha.png')

# # search cosine_alpha
# accs = []
# for cosine_alpha_ in tqdm(cosine_alphas):
#     matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
#                                           mode=mode, dist_alpha=dist_alpha, cosine_alpha=cosine_alpha_,
#                                           ratio_thresh=ratio_thresh, spatial_cosine_thresh=spatial_cosine_thresh, 
#                                           spatial_dist_thresh=spatial_dist_thresh)
#     num_pts_to_evaluate = 100
#     acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
#                                      x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
#                                      x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
#     accs.append(acc)
    
# df = pd.DataFrame({'cosine_alpha': cosine_alphas, 'acc': accs})

# (
#     ggplot(df, aes(x='cosine_alpha', y='acc')) +
#     geom_point() +
#     geom_line() +
#     labs(title='Accuracy vs Cosine Alpha', x='Cosine Alpha', y='Accuracy') +
#     theme_seaborn()
# ).save('images/accuracy_vs_cosine_alpha.png')

# # search ratio_thresh
# accs = []
# for ratio_thresh_ in tqdm(ratio_threshs):
#     matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
#                                           mode=mode, dist_alpha=dist_alpha, cosine_alpha=cosine_alpha,
#                                           ratio_thresh=ratio_thresh_, spatial_cosine_thresh=spatial_cosine_thresh, 
#                                           spatial_dist_thresh=spatial_dist_thresh)
#     num_pts_to_evaluate = 100
#     acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
#                                      x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
#                                      x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
#     accs.append(acc)
    
# df = pd.DataFrame({'ratio_thresh': ratio_threshs, 'acc': accs})
    
# (
#     ggplot(df, aes(x='ratio_thresh', y='acc')) +
#     geom_point() +
#     geom_line() +
#     labs(title='Accuracy vs Ratio Threshold', x='Ratio Threshold', y='Accuracy') +
#     theme_seaborn()
# ).save('images/accuracy_vs_ratio_thresh.png')

# # search spatial_cosine_thresh
# accs = []
# for spatial_cosine_thresh_ in tqdm(spatial_cosine_threshs):
#     matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
#                                           mode=mode, dist_alpha=dist_alpha, cosine_alpha=cosine_alpha,
#                                           ratio_thresh=ratio_thresh, spatial_cosine_thresh=spatial_cosine_thresh_, 
#                                           spatial_dist_thresh=spatial_dist_thresh)
#     num_pts_to_evaluate = 100
#     acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
#                                      x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
#                                      x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
#     accs.append(acc)
    
# df = pd.DataFrame({'spatial_cosine_thresh': spatial_cosine_threshs, 'acc': accs})
        
# (
#     ggplot(df, aes(x='spatial_cosine_thresh', y='acc')) +
#     geom_point() +
#     geom_line() +
#     labs(title='Accuracy vs Spatial Cosine Threshold', x='Spatial Cosine Threshold', y='Accuracy') +
#     theme_seaborn()
# ).save('images/accuracy_vs_spatial_cosine_thresh.png')

# # search spatial_dist_thresh
# accs = []
# for spatial_dist_thresh_ in tqdm(spatial_dist_threshs):
#     matches, confidences = match_features(image1_features, image2_features, x1, y1, x2, y2,
#                                           mode=mode, dist_alpha=dist_alpha, cosine_alpha=cosine_alpha,
#                                           ratio_thresh=ratio_thresh, spatial_cosine_thresh=spatial_cosine_thresh, 
#                                           spatial_dist_thresh=spatial_dist_thresh_)
#     num_pts_to_evaluate = 100
#     acc, c = evaluate_correspondence(image1, image2, eval_file, scale_factor, 
#                                      x1[matches[:num_pts_to_evaluate, 0]], y1[matches[:num_pts_to_evaluate, 0]], 
#                                      x2[matches[:num_pts_to_evaluate, 1]], y2[matches[:num_pts_to_evaluate, 1]])
#     accs.append(acc)
    
# df = pd.DataFrame({'spatial_dist_thresh': spatial_dist_threshs, 'acc': accs})
    
# (
#     ggplot(df, aes(x='spatial_dist_thresh', y='acc')) +
#     geom_point() +
#     geom_line() +
#     labs(title='Accuracy vs Spatial Distance Threshold', x='Spatial Distance Threshold', y='Accuracy') +
#     theme_seaborn()
# ).save('images/accuracy_vs_spatial_dist_thresh.png')
