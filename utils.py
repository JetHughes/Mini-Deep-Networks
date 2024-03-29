import numpy as np
def balanced_accuracy(test_labels, predictions, outputs, name="balanced accuracy"):    
    # https://github.com/rois-codh/kmnist/pull/14/files/2b063db29fc7aa784a60ba69e13f72db892bf435
    totals = []
    for cls in range(outputs):
        total = 0
        for i in test_labels:
            if i == cls:
                total = total + 1
        totals.append(total)

    hits = []
    for cls in range(outputs):
        total_hits = 0
        for i in range(0, test_labels.shape[0]):
            if test_labels[i] == cls == np.argmax(predictions[i]):
                total_hits = total_hits + 1
        hits.append(total_hits)

    accuracy_list = []
    for i in range(0, len(hits)):
        accuracy = hits[i] / totals[i]
        accuracy_list.append(accuracy)

    print(f'The {name} is: {np.mean(accuracy_list)}')