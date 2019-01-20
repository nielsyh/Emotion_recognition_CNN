import matplotlib.pyplot as plt
import numpy as np


epoch_counter = 0

files = ['0.001sgd_20epoch_shape50_37714train_4190test_3cnn_2fc_dropout25.txt',
         'sgd_20epoch_shape50_37714train_4190test_3cnn_2fc_dropout25.txt',
         'adam optimizer.txt',
         '0.001_adam_20epochs.txt',
         'Tanh_Adam_NoIniti.txt',
         'Adam_Tanh_WeightsNormalInitialization.txt',
         'Tanh_Adam_Batch.txt'
         ]


def get_train_data(file):

    accs = []
    times = []
    losses = []

    with open('results/'+file) as f:
        for line in f:
            if 'val_acc' in line:
                # print(line)
                vals = line.split(' ')
                # print(vals)
                times.append(float(vals[3][:-1]))
                losses.append(float(vals[7]))
                accs.append(float(vals[10]))
                # times.append(float(vals[3][:-1]))
                # losses.append(float(vals[13]))
                # accs.append(float(vals[16]))
                # print(vals[3][:-1])
                # print(vals[13])
                # print(vals[16])

    return accs, losses, times

def get_val_data(file):

    accs = []
    times = []
    losses = []

    with open('results/'+file) as f:
        for line in f:
            if 'val_acc' in line:
                # print(line)
                vals = line.split(' ')
                # print(vals)
                # times.append(float(vals[3][:-1]))
                # losses.append(float(vals[7]))
                # accs.append(float(vals[10]))
                times.append(float(vals[3][:-1]))
                losses.append(float(vals[13]))
                accs.append(float(vals[16]))
                # print(vals[3][:-1])
                # print(vals[13])
                # print(vals[16])

    return accs, losses, times

#k15, k16,
def plot_compare(a, b, c,d, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    x = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0]

    plt.plot(x,a, label='SGD lr 0.01', linestyle='-')
    plt.plot(x,b, label='SGD lr 0.001', linestyle='--')
    plt.plot(x,c, label='Adam lr 0.01', linestyle=':')
    plt.plot(x,d, label='Adam lr 0.001', linestyle='-.')


    plt.legend()

    plt.show()

def plot_compare_for_chiara(a, b, c, title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
         20.0]

    plt.plot(x, a, label='TANH NO INIT', linestyle='-')
    plt.plot(x, b, label='TANH GORDON NORM.', linestyle='--')
    plt.plot(x, c, label='TANH BATCH NORM.', linestyle=':')
    plt.legend()

    plt.show()

def plot_test_train(a,b,a_name, b_name,title, label_y, label_x):
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)

    x = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0,19.0,20.0]

    plt.plot(x,a, label=a_name, linestyle='-')
    plt.plot(x,b, label=b_name, linestyle='--')
    plt.legend()

    plt.show()


#plot train data
acc_0, loss_0, time_0 = get_train_data(files[0])
acc_1, loss_1, time_1 = get_train_data(files[1])
acc_2, loss_2, time_2 = get_train_data(files[2])
acc_3, loss_3, time_3 = get_train_data(files[3])

plot_compare(acc_0,acc_1,acc_2,acc_3, 'Train Accuracy', 'Accuracy', 'epoch')
plot_compare(loss_0,loss_1,loss_2, loss_3, 'Train Loss', 'loss', 'epoch')

# #plot tst data
acc_00, loss_00, time_00 = get_val_data(files[0])
acc_11, loss_11, time_11 = get_val_data(files[1])
acc_22, loss_22, time_22 = get_val_data(files[2])
acc_33, loss_33, time_33 = get_val_data(files[3])

plot_compare(acc_00,acc_11,acc_22,acc_33, 'Test Accuracy', 'Accuracy', 'Epoch')
plot_compare(loss_00,loss_11,loss_22, loss_33, 'Test Loss', 'Loss', 'Epoch')

# #compare
plot_test_train(acc_0,acc_00, 'Train sgd lr = 0.01', 'Test sgd lr = 0.01','Train vs Test accuracy', 'Accuracy', 'Epoch')
plot_test_train(acc_1,acc_11, 'Train sgd lr = 0.001', 'Test sgd lr = 0.001','Train vs Test accuracy', 'Accuracy', 'Epoch')

plot_test_train(acc_2,acc_22, 'Train Adam lr = 0.01', 'Test Adam lr = 0.001','Train vs Test accuracy', 'Accuracy', 'Epoch')
plot_test_train(acc_3,acc_33, 'Train Adam lr = 0.001', 'Test Adam lr = 0.001','Train vs Test accuracy', 'Accuracy', 'Epoch')


acc_4, loss_4, time_4 = get_train_data(files[4])
acc_5, loss_5, time_5 = get_train_data(files[5])
acc_6, loss_6, time_6 = get_train_data(files[6])

acc_44, loss_44, time_44 = get_val_data(files[4])
acc_55, loss_55, time_55 = get_val_data(files[5])
acc_66, loss_66, time_55 = get_val_data(files[6])

plot_compare_for_chiara(acc_4,acc_5,acc_6, 'Train Accuracy' , 'Accuracy', 'Epochs')
plot_compare_for_chiara(acc_44,acc_55,acc_66, 'Test Accuracy' , 'Accuracy', 'Epochs')