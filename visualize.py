import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys, os

pkl_loc = sys.argv[1]
with open(pkl_loc, "rb") as f:
    adv_data_dict = pickle.load(f)

xs = adv_data_dict["xs"]
y_trues = adv_data_dict["y_trues"]
y_preds = adv_data_dict["y_preds"]
noises  = adv_data_dict["noises"]
y_preds_outputs = adv_data_dict["y_preds_outputs"]
y_preds_adversarial = adv_data_dict["y_preds_adversarial"]


# import ipdb; ipdb.set_trace()
# visualize N random images
# idxs = np.random.choice(range(50), size=(20,), replace=False)
# for matidx, idx in enumerate(idxs):
idx = 30
print(y_preds[idx])
print(y_trues[idx])
import math
sum = 0.
for tmp in y_preds_outputs[idx][0]:
    sum += math.exp(tmp)
y_prob = []
for tmp in y_preds_outputs[idx][0]:
    y_prob.append(math.exp(tmp) / sum)
for tmp in y_prob:
    print("%.3f %%" % (100. * tmp))
print(y_preds_adversarial[idx])
# print("%d %d" % (matidx, idx))
orig_im = xs[idx].reshape(28,28)
nos_im = noises[idx].reshape(28,28)
rgb_img = np.zeros((28, 28, 3), dtype=np.uint8)
for i in range(28):
    for j in range(28):
        rgb_img[i][j][0] = int(256*nos_im[i][j])
        rgb_img[i][j][1] = int(256*nos_im[i][j])
        rgb_img[i][j][2] = int(256*nos_im[i][j])
        B = rgb_img[i][j][0]
        G = rgb_img[i][j][1]
        R = rgb_img[i][j][2]
        if B <= 51:
            rgb_img[i][j][0] = 255
        elif B <= 102:
            rgb_img[i][j][0] = 255 - (B-51)*5
        elif B <= 153:
            rgb_img[i][j][0] = 0
        else : rgb_img[i][j][0] = 0

        if G <= 51:
            rgb_img[i][j][1] = G*5
        elif G <= 102:
            rgb_img[i][j][1] = 255
        elif G <= 153:
            rgb_img[i][j][1] = 255
        elif G <= 204:
            rgb_img[i][j][1] = 255 - int(128.0*(G-153.0)/51.0 + 0.5)
        else : rgb_img[i][j][1] = 127 - int(127.0*(G-204.0)/51.0 + 0.5)

        if R <= 51:
            rgb_img[i][j][2] = 0
        elif R <= 102:
            rgb_img[i][j][2] = 0
        elif R <= 153:
            rgb_img[i][j][2] = (R-102)*5
        elif G <= 204:
            rgb_img[i][j][2] = 255
        else : rgb_img[i][j][2] = 255

adv_im  = orig_im + noises[idx].reshape(28,28)
# disp_im = np.concatenate((orig_im, noises[idx].reshape(28,28), adv_im), axis=1)
#plt.imshow(orig_im, "gray")
plt.imshow(rgb_img)
#plt.imshow(adv_im, "gray")
plt.xticks([])
plt.yticks([])
plt.show()

# Noise statistics
# import ipdb; ipdb.set_trace()
noises, xs, y_trues, y_preds = np.array(noises), np.array(xs), np.array(y_trues), np.array(y_preds)
noises = noises.squeeze(axis=1)
xs = xs.squeeze(axis=1)
adv_exs = xs + noises
print("Adv examples: max, min: ", adv_exs.max(), adv_exs.min())
print("Noise: Mean, Max, Min: ")
print(np.mean(noises), np.max(noises), np.min(noises))
