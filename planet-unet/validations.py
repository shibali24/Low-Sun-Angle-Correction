import numpy as np
from scipy import ndimage
import pickle, sys
import matplotlib.pyplot as plt
from skimage import filters
from sklearn.metrics import r2_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rc('font', family='serif', size=10)

#############
## some specs
PIXEL_SIZE = 3
ACCEPTED_PROB = 0.75

THRESH_FIXED = 0.20
#################
## some functions
def get_label_stats(mask):
    label_im, nb_labels = ndimage.label(mask)
    sizes_each = ndimage.sum(mask, label_im, range(nb_labels + 1) )
    sizes_sum  = sizes_each.sum()

    return nb_labels, sizes_each, sizes_sum


def plot_comparisons(true, predicted, SAVE_NAME, xlim=None, title=None,
                                      xlabel=None, ylabel=None,
                                      **scatter_args):

    if xlim is None:
        xlim = max(np.asarray(predicted).max(), np.asarray(true).max())
        r2 = r2_score(predicted, true)
    else:
        true = np.where(predicted<xlim, true, np.nan)
        predicted = np.where(predicted<xlim, predicted, np.nan)

        r2 = r2_score(true[~np.isnan(true)],
                      predicted[~np.isnan(predicted)])

    diag = [0,xlim]
    plt.plot(diag, diag, c='grey', ls='--', lw=0.5)

    props = dict(facecolor='w', alpha=0.7)

    plt.text(0.15, 0.91, '$R^2$={:.2f}'.format(r2),
            ha='center', transform=plt.gca().transAxes,
            fontsize=12, bbox=props)

    plt.scatter(true, predicted, **scatter_args)

    plt.gca().set_aspect("equal")
    
    plt.grid(linestyle='dotted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0,xlim])
    plt.ylim([0,xlim])
    plt.title(title)
    plt.xticks(rotation=0)

    plt.savefig(SAVE_NAME, dpi=300)
    plt.clf()

####################################
## validation data
data = pickle.load(open('XXX-SRB-validation.p', 'rb'))
keys = list(data.keys())

labels = []
preds = []
for key in keys:
    lab = key.split('.')
    lab[-2] += '_mask'
    lab = '.'.join(lab)
    labels.append(lab)

    pred = data.get(key)[...,1]
    pred = np.where(pred>ACCEPTED_PROB, 1, 0)
    preds.append(pred)

####################################
## true and pred and otsu statistics
all_counts_true, all_counts_pred, all_counts_otsu, all_counts_thresh = [], [], [], []

all_area_true, all_area_pred, all_area_otsu, all_area_thresh = [], [], [], []
for i in range(len(preds)):
    image_grey = plt.imread(keys[i])[...,0]
    thresh_otsu = filters.threshold_otsu(image_grey)

    otsu_count, _, otsu_area = get_label_stats(image_grey > thresh_otsu)
    true_count, _, true_area = get_label_stats(plt.imread(labels[i]))
    pred_count, _, pred_area = get_label_stats(preds[i])
    thresh_count, _, thresh_area = get_label_stats(image_grey > THRESH_FIXED)
    
    all_counts_true.append(true_count)
    all_counts_pred.append(pred_count)
    all_counts_otsu.append(otsu_count)
    all_counts_thresh.append(thresh_count)

    all_area_true.append(true_area*PIXEL_SIZE**2)
    all_area_pred.append(pred_area*PIXEL_SIZE**2)
    all_area_otsu.append(otsu_area*PIXEL_SIZE**2)
    all_area_thresh.append(thresh_area*PIXEL_SIZE**2)


all_area_true = np.asarray(all_area_true)
all_area_pred = np.asarray(all_area_pred)

all_area_otsu = np.asarray(all_area_otsu)
all_area_thresh = np.asarray(all_area_thresh)

############
## all plots
args_area = {'c':'b',
             's':25,
             'edgecolors':'w',
             'alpha':0.7}

args_count = {'c':'r',
              's':25,
              'edgecolors':'k',
              'alpha':0.7}

fixed_xlim = max(all_area_pred.max()/1e6, all_area_true.max()/1e6)

## comparison plots for total area
plt.xticks(np.arange(0, 0.42, 0.05))
plt.yticks(np.arange(0, 0.42, 0.05))
plot_comparisons(true=all_area_true/1e6, predicted=all_area_pred/1e6,
                 SAVE_NAME='./figs-manuscript/validation-p-unet-area2.png',
                 xlim=None, title='Area: True vs. UNet',
                 xlabel=r'True iceberg net area [km$^2$]',
                 ylabel=r'UNet iceberg net area [km$^2$]',
                 **args_area)

plt.xticks(np.arange(0, 0.42, 0.05))
plt.yticks(np.arange(0, 0.42, 0.05))
plot_comparisons(true=all_area_true/1e6, predicted=all_area_otsu/1e6, xlim = fixed_xlim,
                 SAVE_NAME='./figs-manuscript/validation-p-otsu-area2.png',
                 title='Area: True vs. Otsu',
                 xlabel=r'True iceberg net area [km$^2$]',
                 ylabel=r'Otsu iceberg net area [km$^2$]', **args_area)

plt.xticks(np.arange(0, 0.42, 0.05))
plt.yticks(np.arange(0, 0.42, 0.05))
plot_comparisons(true=all_area_true/1e6, predicted=all_area_thresh/1e6, xlim = fixed_xlim,
                 SAVE_NAME='./figs-manuscript/validation-p-thresh-area2.png',
                 title='Area: True vs. simple thresholding',
                 xlabel=r'True iceberg net area [km$^2$]',
                 ylabel=r'Simple thresholding iceberg net area [km$^2$]', **args_area)


## comparison plots for total iceberg count
## minor modification to crop out sub-images with rocks
mod_all_counts_true = np.where(all_area_otsu<fixed_xlim*1e6, all_counts_true, np.nan)
mod_all_counts_true = mod_all_counts_true[~np.isnan(mod_all_counts_true)]

mod_all_counts_pred = np.where(all_area_otsu<fixed_xlim*1e6, all_counts_pred, np.nan)
mod_all_counts_pred = mod_all_counts_pred[~np.isnan(mod_all_counts_pred)]

mod_all_counts_otsu = np.where(all_area_otsu<fixed_xlim*1e6, all_counts_otsu, np.nan)
mod_all_counts_otsu = mod_all_counts_otsu[~np.isnan(mod_all_counts_otsu)]

mod_all_counts_thresh = np.where(all_area_otsu<fixed_xlim*1e6, all_counts_thresh, np.nan)
mod_all_counts_thresh = mod_all_counts_thresh[~np.isnan(mod_all_counts_thresh)]

plot_comparisons(true=mod_all_counts_true, predicted=mod_all_counts_pred,
                 SAVE_NAME='./figs-manuscript/validation-p-unet-count2.png',
                 xlim=250, title='Count: True vs. UNet',
                 xlabel='True iceberg count',
                 ylabel='UNet iceberg count', **args_count)

plot_comparisons(true=mod_all_counts_true, predicted=mod_all_counts_otsu,
                 SAVE_NAME='./figs-manuscript/validation-p-otsu-count2.png',
                 xlim=250, title='Count: True vs. Otsu',
                 xlabel='True iceberg count',
                 ylabel='Otsu iceberg count', **args_count)

plot_comparisons(true=mod_all_counts_true, predicted=mod_all_counts_thresh,
                 SAVE_NAME='./figs-manuscript/validation-p-thresh-count2.png',
                 xlim=250, title='Count: True vs. simple thresholding',
                 xlabel='True iceberg count',
                 ylabel='Simple thresholding iceberg count', **args_count)

