import numpy as np
data_dir = '/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/14/'
def calculate_l2_distance(predictions):
    targets = np.load(data_dir + 'TEST_LABEL.npy').astype(np.float32)
    predictions = predictions.astype(np.float32)
    targets = targets[1:,:]
    predictions = predictions[1:,:]
    targets = targets * 224.
    predictions = predictions * 224.

    dist = np.power((predictions-targets),2)
    dist = np.sum(dist,axis=1)
    dist = np.sqrt(dist)
    dist = np.mean(dist)

    return dist




def evaluate_detailed(predictions):
    predictions = predictions * 224.
    if data_dir == '/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/14/' or data_dir == '/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTS300/15/'or data_dir == '/mnt/e/data/Improved-TCLNet/Improved-TCLNet/datasets/data/TCLDwithTSsame/15/':
        targets = np.load(data_dir + 'TEST_LABEL.npy').astype(np.float32)*224.
        points1 = np.load(data_dir + 'Index2.npy')
        points2 = np.load(data_dir + "Index3.npy")
        points3 = np.load(data_dir + "Index4.npy")
        points4 = np.load(data_dir + "Index5.npy")
        points5 = np.load(data_dir + "Index6.npy")
        points6 = np.load(data_dir + "Index7.npy")

        mean1 = 0.0
        mean2 = 0.0
        mean3 = 0.0
        mean4 = 0.0
        mean5 = 0.0
        mean6 = 0.0
        mean_overall = 0.0

        for i in range(len(points1)):
            mean_easypont = np.power((targets[points1[i]] - predictions[points1[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean1 = mean1 + mean_easypont
        for i in range(len(points2)):
            mean_easypont = np.power((targets[points2[i]] - predictions[points2[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean2 = mean2 + mean_easypont
        for i in range(len(points3)):
            mean_easypont = np.power((targets[points3[i]] - predictions[points3[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean3 = mean3 + mean_easypont
        for i in range(len(points4)):
            mean_easypont = np.power((targets[points4[i]] - predictions[points4[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean4 = mean4 + mean_easypont
        for i in range(len(points5)):
            mean_easypont = np.power((targets[points5[i]] - predictions[points5[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean5 = mean5 + mean_easypont
        for i in range(len(points6)):
            mean_hardpoint = np.power((targets[points6[i]] - predictions[points6[i]]), 2)
            mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
            mean6 = mean6 + mean_hardpoint

        print(targets[0].shape,predictions[0].shape)
        for k in range(3200):
            # print(targets[k+1],predictions[k+1],targets[k+1] - predictions[k+1])
            mean_overpoint = np.power((targets[k+1] - predictions[k+1]), 2)
            mean_overpoint = np.sqrt(np.sum(mean_overpoint))
            mean_overall = mean_overall + mean_overpoint

        mean1 = mean1 / len(points1)
        mean2 = mean2 / len(points2)
        mean3 = mean3 / len(points3)
        mean4 = mean4 / len(points4)
        mean5 = mean5 / len(points5)
        mean6 = mean6 / len(points6)
        mean_overall = mean_overall / (len(predictions)-1)

        return mean_overall,mean1,mean2,mean3,mean4,mean5,mean6
    else:
        targets = np.load(data_dir + 'TEST_LABEL.npy').astype(np.float32) * 224.
        # points1 = np.load(data_dir + 'Index2.npy')
        points2 = np.load(data_dir + "Index3.npy")
        points3 = np.load(data_dir + "Index4.npy")
        points4 = np.load(data_dir + "Index5.npy")
        points5 = np.load(data_dir + "Index6.npy")
        points6 = np.load(data_dir + "Index7.npy")

        mean1 = 0.0
        mean2 = 0.0
        mean3 = 0.0
        mean4 = 0.0
        mean5 = 0.0
        mean6 = 0.0
        mean_overall = 0.0

        # for i in range(len(points1)):
        #     mean_easypont = np.power((targets[points1[i]] - predictions[points1[i]]), 2)
        #     mean_easypont = np.sqrt(np.sum(mean_easypont))
        #     mean1 = mean1 + mean_easypont
        for i in range(len(points2)):
            mean_easypont = np.power((targets[points2[i]] - predictions[points2[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean2 = mean2 + mean_easypont
        for i in range(len(points3)):
            mean_easypont = np.power((targets[points3[i]] - predictions[points3[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean3 = mean3 + mean_easypont
        for i in range(len(points4)):
            mean_easypont = np.power((targets[points4[i]] - predictions[points4[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean4 = mean4 + mean_easypont
        for i in range(len(points5)):
            mean_easypont = np.power((targets[points5[i]] - predictions[points5[i]]), 2)
            mean_easypont = np.sqrt(np.sum(mean_easypont))
            mean5 = mean5 + mean_easypont
        for i in range(len(points6)):
            mean_hardpoint = np.power((targets[points6[i]] - predictions[points6[i]]), 2)
            mean_hardpoint = np.sqrt(np.sum(mean_hardpoint))
            mean6 = mean6 + mean_hardpoint

        print(targets.shape, predictions.shape)
        for k in range(len(predictions) - 1):
            mean_overpoint = np.power((targets[k + 1] - predictions[k + 1]), 2)
            mean_overpoint = np.sqrt(np.sum(mean_overpoint))
            mean_overall = mean_overall + mean_overpoint

        # mean1 = mean1 / len(points1)
        mean2 = mean2 / len(points2)
        mean3 = mean3 / len(points3)
        mean4 = mean4 / len(points4)
        mean5 = mean5 / len(points5)
        mean6 = mean6 / len(points6)
        mean_overall = mean_overall / (len(predictions) - 1)

        return mean_overall, mean1, mean2, mean3, mean4, mean5, mean6