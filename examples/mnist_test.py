import argparse
import numpy as np
import cv2
from toolbox import dataframe
from toolbox.utils import train_test_split, print_pred_vs_obs,\
                          pred_accuracy, pred_mean_error, np_gray2rgb
from neurals_network.mlp import MlpClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('csvfile', help='data.csv')
    parser.add_argument('model', help='trained model')
    parser.add_argument('output_label', help='name of labels column')
    args = parser.parse_args()

    return args

def main():
    displaying_mode = "all"  # must be in ("all", "only_false", "only_true")
    args = parse_arguments()
    output_label = args.output_label
    mlp = MlpClassifier.load(args.model)
    df = dataframe.DataFrame.read_csv(args.csvfile)
    df.set_numerical_features(to_remove=[output_label])
    df.digitalize()
    X = [x for feature, x in df.data.items() if feature in df.numerical_features]
    y = df.data[output_label]
    _, _, X_test, y_test = train_test_split(X, y, train_ratio=0.)
    pred = mlp.predict(X_test)
    y_test = [y[0] for y in y_test.T.tolist()]
    print_pred_vs_obs(pred, y_test, only_false=False)
    print("\naccuracy: {}%".format(100*round(pred_accuracy(pred, y_test), 5)))
    if displaying_mode == "all":
        index_to_display = range(len(y_test))
    elif displaying_mode == "only_false":
        index_to_display = [i for i in range(len(y_test)) if pred[i] != y_test[i]]
    elif displaying_mode == "only_true":
        index_to_display = [i for i in range(len(y_test)) if pred[i] == y_test[i]]
    else:
        raise Exception("unknown displaying mode")
    nb_samples = X_test.shape[1]
    nb_to_display = len(index_to_display)
    img_size = 1000
    digit_size = 400
    ratio = float(img_size) / 500.
    police = 1. * ratio
    i = 0
    while True:
        img = np.zeros((img_size, img_size, 3))
        index = index_to_display[i]
        digit = X_test[:, index].reshape((28, 28))
        digit = np_gray2rgb(digit)
        digit = cv2.resize(digit, (digit_size, digit_size))
        color_real = (255, 255, 255)
        color_pred = (0, 255, 0) if y_test[index] == pred[index] else (0, 0, 255)
        img[int((img_size - digit_size) / 2):int((img_size + digit_size) / 2),
            int((img_size - digit_size) / 2):int((img_size + digit_size) / 2),
        ] = digit
        cv2.putText(img, '{:<10}: {}'.format('real', int(y_test[index])),
            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, police,
            color_real, thickness=1)
        cv2.putText(img, '{:<10}: {}'.format('predicted', int(pred[index])),
            (10, 70 + int(40 * ratio)), cv2.FONT_HERSHEY_SIMPLEX, police,
            color_pred, thickness=1)
        cv2.imshow("digit", img)
        while True:
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
            elif key & 0xFF == 81 and i > 0:
                i -= 1
                break
            elif key & 0xFF == 83 and i < nb_to_display - 1:
                i += 1
                break
    return

if __name__ == '__main__':
    main()
