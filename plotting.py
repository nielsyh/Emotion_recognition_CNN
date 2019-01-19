import matplotlib.pyplot as plt


epoch_counter = 0

files = ['0.001sgd_20epoch_shape50_37714train_4190test_3cnn_2fc_dropout25.txt',
         'sgd_20epoch_shape50_37714train_4190test_3cnn_2fc_dropout25.txt'
         ]


def get_data(file):

    accs = []
    times = []
    losses = []

    with open('results/'+file) as f:
        for line in f:
            if 'val_acc' in line:
                # print(line)
                vals = line.split(' ')
                times.append(float(vals[3][:-1]))
                losses.append(float(vals[13]))
                accs.append(float(vals[16]))
                # print(vals[3][:-1])
                # print(vals[13])
                # print(vals[16])

    return accs, losses, times


#k15, k16,
def plot_stuff(a, b, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)


    plt.plot(a, label='lr 0.01', linestyle='-')
    plt.plot(b, label='lr 0.001', linestyle='--')
    # plt.plot(k16, label='16', linestyle=':')
    # plt.plot(k15, label='15', linestyle='-.')

    plt.legend()

    plt.show()


#model 1
acc_0, loss_0, time_0 = get_data(files[0])
acc_1, loss_1, time_1 = get_data(files[1])

plot_stuff(acc_0,acc_1, 'SGD accuracy', 'Accuracy', 'epoch')
plot_stuff(time_0,time_1, 'SGD processing-time', 'Seconds', 'epoch')
plot_stuff(loss_0,loss_1, 'SGD Loss', 'loss', 'epoch')