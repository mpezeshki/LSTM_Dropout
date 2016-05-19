import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


colors = [(31 / 255.0, 119 / 255.0, 180 / 255.0),
          (174 / 255.0, 199 / 255.0, 232 / 255.0),
          (255 / 255.0, 127 / 255.0, 14 / 255.0),
          (255 / 255.0, 187 / 255.0, 120 / 255.0),
          (44 / 255.0, 160 / 255.0, 44 / 255.0),
          (152 / 255.0, 223 / 255.0, 138 / 255.0),
          (214 / 255.0, 39 / 255.0, 40 / 255.0),
          (255 / 255.0, 152 / 255.0, 150 / 255.0),
          (148 / 255.0, 103 / 255.0, 189 / 255.0),
          (197 / 255.0, 176 / 255.0, 213 / 255.0),
          (140 / 255.0, 86 / 255.0, 75 / 255.0),
          (196 / 255.0, 156 / 255.0, 148 / 255.0),
          (227 / 255.0, 119 / 255.0, 194 / 255.0),
          (247 / 255.0, 182 / 255.0, 210 / 255.0),
          (127 / 255.0, 127 / 255.0, 127 / 255.0),
          (199 / 255.0, 199 / 255.0, 199 / 255.0),
          (188 / 255.0, 189 / 255.0, 34 / 255.0),
          (219 / 255.0, 219 / 255.0, 141 / 255.0),
          (23 / 255.0, 190 / 255.0, 207 / 255.0),
          (158 / 255.0, 218 / 255.0, 229 / 255.0)]


def parse_log(path, to_be_plotted):
    results = {}
    log = open(path, 'r').readlines()
    for line in log:
        colon_index = line.find(":")
        enter_index = line.find("\n")
        if colon_index != -1:
            key = line[:colon_index]
            value = line[colon_index + 1: enter_index]
            if key in to_be_plotted:
                values = results.get(key)
                if values is None:
                    results[key] = [value]
                else:
                    results[key] = results[key] + [value]
    for key in results.keys():
        results[key] = [float(i) for i in results[key]]
    return results


def pimp(path=None, xaxis='Epochs', yaxis='Cross Entropy', title=None):
    plt.legend(fontsize=14)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.grid()
    plt.title(title)
    plt.ylim([0, 1.0])
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def plot(x, y, xlabel='train', ylabel='dev', color='b',
         x_steps=None, y_steps=None):
    if x_steps is None:
        x_steps = range(len(x))
    if y_steps is None:
        y_steps = range(len(y))
    plt.plot(x_steps, x, ls=':', c=color, lw=2, label=xlabel)
    plt.plot(y_steps, y, c=color, lw=2, label=ylabel)


def best(path, what='valid_error_rate'):
    res = parse_log(path, [what])
    return np.min([float(i) for i in res[what]])


to_be_plotted = ['train_evaluation_error_rate', 'valid_error_rate']
yaxis = 'Error Rate'
main_title = ''

files = ['LSTM_LSTM_2',
         'LSTM_ZoneOut_2',
         'LSTM_Elephant_2']
path = '/u/pezeshki/LSTM_Dropout/'

plt.figure()
for i, file in enumerate(files):
    log = path + file + '/log.txt'
    results = parse_log(log, to_be_plotted)
    plot(results[to_be_plotted[0]], results[to_be_plotted[1]],
         'train ' + file, 'valid ' + file, colors[i])
    print 'Best of valid in model ' + file + ': ' + str(best(log, 'valid_error_rate'))

pimp(path=None, yaxis=yaxis, title=main_title)
plt.savefig(path + 'plot.png', dpi=300)
