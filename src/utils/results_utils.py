import h5py
import numpy as np
import pandas as pd 
import os
import numpy as np
import re
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def convert_csv_to_txt(input_file,output_file):
   
    with open(input_file, 'r') as csv_file, open(output_file, 'w') as space_delimited_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            space_delimited_file.write(' '.join(row) + '\n')

    print(f'CSV file "{input_file}" converted to space-delimited file "{output_file}"')



def read_file(file):
    hf = h5py.File(file, 'r')
    attributes = []
    for key in hf.keys():
        attributes.append(key)
    
    return attributes, hf


def get_data(hf,attributes):
    data = []
    pm = []
    acc_pm = []
    loss_pm = []
    loss_gm = []
    for i in range(len(attributes)):
        ai = hf.get(attributes[i])
        ai = np.array(ai)
        data.append(ai)
    
    return data

def convergence_analysis(path, acc_file, loss_file):
    dir_list = os.listdir(path)
    
    Fedavg_gd_test_loss = []
    Fedavg_gd_test_accuracy = []
    Fedavg_gd_train_loss = []
    Fedavg_gd_train_accuracy = []

    Fedavg_sgd_test_loss = []
    Fedavg_sgd_test_accuracy = []
    Fedavg_sgd_train_loss = []
    Fedavg_sgd_train_accuracy = []

    Fedavg_pgd_test_loss = []
    Fedavg_pgd_test_accuracy = []
    Fedavg_pgd_train_loss = []
    Fedavg_pgd_train_accuracy = []
    
    Fedavg_psgd_test_loss = []
    Fedavg_psgd_test_accuracy = []
    Fedavg_psgd_train_loss = []
    Fedavg_psgd_train_accuracy = []

    Fedprox_test_loss = []
    Fedprox_test_accuracy = []
    Fedprox_train_loss = []
    Fedprox_train_accuracy = []

    FedFW_cel_test_loss = []
    FedFW_cel_test_accuracy = []
    FedFW_cel_train_loss = []
    FedFW_cel_train_accuracy = []

    FedFW_tel_test_loss = []
    FedFW_tel_test_accuracy = []
    FedFW_tel_train_loss = []
    FedFW_tel_train_accuracy = []


    for file_name in dir_list:
        if file_name in ['fedavg_gd.h5','fedavg_sgd.h5', 'fedavg_pgd.h5', 'fedavg_psgd.h5', 'fedfw_cel.h5', 'fedfw_tel.h5', 'fedprox.h5']:
            print(file_name)
            attributes, hf = read_file(path+file_name)

            data = get_data(hf,attributes)
            #id=0
            for key in hf.keys():
                attributes.append(key)
                # print("id [",id,"] :", key)
                #id+=1
                
            train_loss = hf.get('avg_training_loss')
            train_acc = hf.get('avg_training_accuracy')   
            val_loss = hf.get('avg_test_loss')
            val_acc = hf.get('avg_test_accuracy')
                
            
            if file_name == "fedavg_gd.h5":
                Fedavg_gd_train_loss.append(np.array(train_loss).tolist())
                Fedavg_gd_train_accuracy.append(np.array(train_acc).tolist())
                Fedavg_gd_test_loss.append(np.array(val_loss).tolist())
                Fedavg_gd_test_accuracy.append(np.array(val_acc).tolist())

            elif file_name == "fedavg_sgd.h5":
                Fedavg_sgd_train_loss.append(np.array(train_loss).tolist())
                Fedavg_sgd_train_accuracy.append(np.array(train_acc).tolist())
                Fedavg_sgd_test_loss.append(np.array(val_loss).tolist())
                Fedavg_sgd_test_accuracy.append(np.array(val_acc).tolist())


            elif file_name == "fedavg_pgd.h5":
                Fedavg_pgd_train_loss.append(np.array(train_loss).tolist())
                Fedavg_pgd_train_accuracy.append(np.array(train_acc).tolist())
                Fedavg_pgd_test_loss.append(np.array(val_loss).tolist())
                Fedavg_pgd_test_accuracy.append(np.array(val_acc).tolist())

                #print(Fedavg_pgd_train_loss)

            elif file_name == "fedavg_psgd.h5":
                Fedavg_psgd_train_loss.append(np.array(train_loss).tolist())
                Fedavg_psgd_train_accuracy.append(np.array(train_acc).tolist())
                Fedavg_psgd_test_loss.append(np.array(val_loss).tolist())
                Fedavg_psgd_test_accuracy.append(np.array(val_acc).tolist())


            elif file_name == "fedprox.h5":

                Fedprox_train_loss.append(np.array(train_loss).tolist())
                Fedprox_train_accuracy.append(np.array(train_acc).tolist())
                Fedprox_test_loss.append(np.array(val_loss).tolist())
                Fedprox_test_accuracy.append(np.array(val_acc).tolist())

            elif file_name == "fedfw_cel.h5":

                FedFW_cel_train_loss.append(np.array(train_loss).tolist())
                FedFW_cel_train_accuracy.append(np.array(train_acc).tolist())
                FedFW_cel_test_loss.append(np.array(val_loss).tolist())
                FedFW_cel_test_accuracy.append(np.array(val_acc).tolist())
            
            elif file_name == "fedfw_tel.h5":

                FedFW_tel_train_loss.append(np.array(train_loss).tolist())
                FedFW_tel_train_accuracy.append(np.array(train_acc).tolist())
                FedFW_tel_test_loss.append(np.array(val_loss).tolist())
                FedFW_tel_test_accuracy.append(np.array(val_acc).tolist())

    train_loss = {
        'GR' : np.arange(1000),
        #'FedavgGD' :  Fedavg_gd_train_loss,
        #'FedavgSGD' :  Fedavg_sgd_train_loss,
        'FedavgPGD' :   Fedavg_pgd_train_loss[0][:],
        'FedavgPSGD' : Fedavg_psgd_train_loss[0][:],
        'Fedprox' : Fedprox_train_loss[0][:],
        'Fedfw_cel' : FedFW_cel_train_loss[0][:],
        'Fedfw_tel' : FedFW_tel_train_loss[0][:],
             
                    }

    train_acc = {
        'GR' : np.arange(1000),
        #'FedavgGD' :  Fedavg_gd_train_accuracy,
        #'FedavgSGD' :  Fedavg_sgd_train_accuracy,
        'FedavgPGD' :   Fedavg_pgd_train_accuracy[0][:],
        'FedavgPSGD' : Fedavg_psgd_train_accuracy[0][:],
        'Fedprox' : Fedprox_train_accuracy[0][:],
        'Fedfw_cel' : FedFW_cel_train_accuracy[0][:],
        'Fedfw_tel' : FedFW_tel_train_accuracy[0][:] 
             
                    }

    val_loss = {
        'GR' : np.arange(1000),
        #'FedavgGD' :  Fedavg_gd_test_loss,
        #'FedavgSGD' :  Fedavg_sgd_test_loss,
        'FedavgPGD' :   Fedavg_pgd_test_loss[0][:],
        'FedavgPSGD' : Fedavg_psgd_test_loss[0][:],
        'Fedprox' : Fedprox_test_loss[0][:],
        'Fedfw_cel' : FedFW_cel_test_loss[0][:],
        'Fedfw_tel' : FedFW_tel_test_loss[0][:] 
             
                    }

    val_acc = {
        'GR' : np.arange(1000),
        #'FedavgGD' :  Fedavg_gd_test_accuracy,
        #'FedavgSGD' :  Fedavg_sgd_test_accuracy,
        'FedavgPGD' :   Fedavg_pgd_test_accuracy[0][:],
        'FedavgPSGD' : Fedavg_psgd_train_accuracy[0][:],
        'Fedprox' : Fedprox_test_accuracy[0][:],
        'Fedfw_cel' : FedFW_cel_test_accuracy[0][:],
        'Fedfw_tel' : FedFW_tel_test_accuracy[0][:] 
                  
                    }

    df_train_loss = pd.DataFrame(train_loss)
    df_train_acc = pd.DataFrame(train_acc)
    df_val_loss = pd.DataFrame(val_loss)
    df_val_acc = pd.DataFrame(val_acc)

    csv_train_acc_path = path + "train_" + acc_file +".csv"
    csv_train_loss_path = path + "train_" + loss_file +".csv"
    csv_val_acc_path = path + "test_" + acc_file +".csv"
    csv_val_loss_path = path + "test_" +loss_file +".csv"
    
    
    txt_train_acc_path = path + "train_" + acc_file +".txt"
    txt_train_loss_path = path +  "train_" + loss_file +".txt"
    txt_val_acc_path = path + "test_" + acc_file +".txt"
    txt_val_loss_path = path + "test_" + loss_file +".txt"

    df_train_acc.to_csv(csv_train_acc_path, index=False)
    df_train_loss.to_csv(csv_train_loss_path, index=False)
    df_val_acc.to_csv(csv_val_acc_path, index=False)
    df_val_loss.to_csv(csv_val_loss_path, index=False)
    
    convert_csv_to_txt(csv_train_acc_path,txt_train_acc_path)
    convert_csv_to_txt(csv_train_loss_path,txt_train_loss_path)
    convert_csv_to_txt(csv_val_acc_path,txt_val_acc_path)
    convert_csv_to_txt(csv_val_loss_path,txt_val_loss_path)

    plot_convergence(#Fedavg_gd_test_loss[0],
                    #Fedavg_gd_test_accuracy[0],
                    #Fedavg_gd_train_loss[0],
                    #Fedavg_gd_train_accuracy[0],
                    #Fedavg_sgd_test_loss[0],
                    #Fedavg_sgd_test_accuracy[0],
                    #Fedavg_sgd_train_loss[0],
                    #Fedavg_sgd_train_accuracy[0],
                    Fedavg_pgd_test_loss[0],
                    Fedavg_pgd_test_accuracy[0],
                    Fedavg_pgd_train_loss[0],
                    Fedavg_pgd_train_accuracy[0],
                    Fedavg_psgd_test_loss[0],
                    Fedavg_psgd_test_accuracy[0],
                    Fedavg_psgd_train_loss[0],
                    Fedavg_psgd_train_accuracy[0],
                    Fedprox_test_loss[0],
                    Fedprox_test_accuracy[0],
                    Fedprox_train_loss[0],
                    Fedprox_train_accuracy[0],
                    FedFW_cel_test_loss[0],
                    FedFW_cel_test_accuracy[0],
                    FedFW_cel_train_loss[0],
                    FedFW_cel_train_accuracy[0],
                    FedFW_tel_test_loss[0],
                    FedFW_tel_test_accuracy[0],
                    FedFW_tel_train_loss[0],
                    FedFW_tel_train_accuracy[0], path)

def plot_convergence(#Fedavg_gd_test_loss,
                    #Fedavg_gd_test_accuracy,
                    #Fedavg_gd_train_loss,
                    #Fedavg_gd_train_accuracy,
                    #Fedavg_sgd_test_loss,
                    #Fedavg_sgd_test_accuracy,
                    #Fedavg_sgd_train_loss,
                    #Fedavg_sgd_train_accuracy,
                    Fedavg_pgd_test_loss,
                    Fedavg_pgd_test_accuracy,
                    Fedavg_pgd_train_loss,
                    Fedavg_pgd_train_accuracy,
                    Fedavg_psgd_test_loss,
                    Fedavg_psgd_test_accuracy,
                    Fedavg_psgd_train_loss,
                    Fedavg_psgd_train_accuracy,
                    Fedprox_test_loss,
                    Fedprox_test_accuracy,
                    Fedprox_train_loss,
                    Fedprox_train_accuracy,
                    FedFW_cel_test_loss,
                    FedFW_cel_test_accuracy,
                    FedFW_cel_train_loss,
                    FedFW_cel_train_accuracy,
                    FedFW_tel_test_loss,
                    FedFW_tel_test_accuracy,
                    FedFW_tel_train_loss,
                    FedFW_tel_train_accuracy, path):
        
        
        fig, ax = plt.subplots(1,4, figsize=(20,4))

       # ax[0].plot(Fedavg_gd_test_loss, label= "FedAvg+GD")
       #ax[0].plot(Fedavg_sgd_test_loss, label= "FedAvg+SGD")
        ax[0].plot(Fedavg_pgd_test_loss, label= "FedAvg+PGD")
        ax[0].plot(Fedavg_psgd_test_loss, label= "FedAvg+PSGD")
        ax[0].plot(Fedprox_test_loss, label= "FedProx")
        ax[0].plot(FedFW_cel_test_loss, label= "FedFW_cel")
        ax[0].plot(FedFW_tel_test_loss, label= "FedFW_tel")
        ax[0].set_xlabel("Global Iteration")
        #ax[0].set_xscale('log')
        ax[0].set_ylabel("Validation Loss")
        #ax[0].set_yscale('log')
        ax[0].set_xticks(range(0, 1000, int(1000/5)))
        #ax[0].legend(prop={"size":12})
        #ax[0].legend()
        x1, x2, y1, y2 = 600, 800, 0.15, 2.0  # Adjust these values as needed
        axins = inset_axes(ax[0], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_pgd_test_loss)
        axins.plot(Fedavg_psgd_test_loss)
        axins.plot(Fedprox_test_loss) 
        axins.plot(FedFW_cel_test_loss) 
        axins.plot(FedFW_tel_test_loss)
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")

       # ax[1].plot(Fedavg_gd_test_accuracy, label= "FedAvg+GD")
        #ax[1].plot(Fedavg_sgd_test_accuracy, label= "FedAvg+SGD")
        ax[1].plot(Fedavg_pgd_test_accuracy, label= "FedAvg+PGD")
        ax[1].plot(Fedavg_psgd_test_accuracy, label= "FedAvg+PSGD")
        ax[1].plot(Fedprox_test_accuracy, label= "FedProx")
        ax[1].plot(FedFW_cel_test_accuracy, label= "FedFW_cel")
        ax[1].plot(FedFW_tel_test_accuracy, label= "FedFW_tel")
        ax[1].set_xlabel("Global Iteration")
        ax[1].set_xticks(range(0, 1000, int(1000/5)))
        ax[1].set_ylabel("Validation Accuracy")
        #ax[1].legend(prop={"size":12})
        #ax[1].legend()
        x1, x2, y1, y2 = 600, 800, 0.8, 0.92  # Adjust these values as needed
        axins = inset_axes(ax[1], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_pgd_test_accuracy)
        axins.plot(Fedavg_psgd_test_accuracy)
        axins.plot(Fedprox_test_accuracy) 
        axins.plot(FedFW_cel_test_accuracy)
        axins.plot(FedFW_tel_test_accuracy) 
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")

        # ax[2].plot(Fedavg_gd_train_loss, label= "FedAvg+GD")
        #ax[2].plot(Fedavg_sgd_train_loss, label= "FedAvg+SGD")
        ax[2].plot(Fedavg_pgd_train_loss , label= "FedAvg+PGD")
        ax[2].plot(Fedavg_psgd_train_loss, label= "FedAvg+PSGD")
        ax[2].plot(Fedprox_train_loss, label= "FedProx")
        ax[2].plot(FedFW_cel_train_loss, label= "FedFW_cel")
        ax[2].plot(FedFW_tel_train_loss, label= "FedFW_tel")
        ax[2].set_xlabel("Global Iteration")
        #ax[2].set_xscale('log')
        ax[2].set_ylabel("Training Loss")
        #ax[2].set_yscale('log')
        ax[2].set_xticks(range(0, 1000, int(1000/5)))
        #ax[2].legend(prop={"size":12})
        #ax[2].legend()
        x1, x2, y1, y2 = 600, 800, 0.1, 0.55  # Adjust these values as needed
        axins = inset_axes(ax[2], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_pgd_train_loss, label="FedAvg+PGD")
        axins.plot(Fedavg_psgd_train_loss, label= "FedAvg+PSGD")
        axins.plot(Fedprox_train_loss, label= "FedProx")
        axins.plot(FedFW_cel_train_loss, label= "FedFW_cel")
        axins.plot(FedFW_tel_train_loss, label= "FedFW_tel")
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")

        # ax[3].plot(Fedavg_gd_train_accuracy, label= "FedAvg+GD")
        #ax[3].plot(Fedavg_sgd_train_accuracy, label= "FedAvg+SGD")
        ax[3].plot(Fedavg_pgd_train_accuracy, label= "FedAvg+PGD")
        ax[3].plot(Fedavg_psgd_train_accuracy, label= "FedAvg+PSGD")
        ax[3].plot(Fedprox_train_accuracy, label= "FedProx")
        ax[3].plot(FedFW_cel_train_accuracy, label= "FedFW_cel")
        ax[3].plot(FedFW_tel_train_accuracy, label= "FedFW_tel")
        ax[3].set_xlabel("Global Iteration")
        ax[3].set_ylabel("Training Accuracy")
        ax[3].set_xticks(range(0, 1000, int(1000/5)))
        #ax[3].legend(prop={"size":12})
        #ax[3].legend()


        x1, x2, y1, y2 = 600, 800, 0.8, 0.98  # Adjust these values as needed
        axins = inset_axes(ax[3], width="50%", height="50%", loc=7)
        axins.plot(Fedavg_pgd_train_accuracy)
        axins.plot(Fedavg_psgd_train_accuracy)
        axins.plot(Fedprox_train_accuracy) 
        axins.plot(FedFW_cel_train_accuracy)
        axins.plot(FedFW_tel_train_accuracy) 
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.indicate_inset_zoom(axins, edgecolor="black")
        handles, labels = ax[3].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=12)


        plt.draw()
       
        plt.savefig(path +'convergence.png')

        # Show the graph
        plt.show()



def average_result(path,directory_name, algorithm, avg_file):
    
    dir_list = os.listdir(path)

    i=0
    train_loss, train_accuracy, test_loss, test_accuracy = [], [], [], []

    if algorithm in [ "FedavgGD", "FedavgSGD", "FedavgPGD", "FedavgPSGD", "Fedprox", "Fedfw" ]:
        for file_name in dir_list:
            
            
            if file_name.endswith(".h5"):
                print(file_name)
                attributes, hf = read_file(path+file_name)

                data = get_data(hf,attributes)
                id=0
                for key in hf.keys():
                    attributes.append(key)
                    # print("id [",id,"] :", key)
                    id+=1

                gtsl = hf.get('global_test_loss')
                gtrl = hf.get('global_train_loss')
                gtsa = hf.get('global_test_accuracy')
                gtra = hf.get('global_train_accuracy')

                test_loss.append(np.array(gtsl).tolist())
                train_loss.append(np.array(gtrl).tolist())
                test_accuracy.append(np.array(gtsa).tolist())
                train_accuracy.append(np.array(gtra).tolist())
   
                
            
        avg_train_loss = np.array(train_loss)
        avg_test_loss = np.array(test_loss)
        avg_train_accuracy = np.array(train_accuracy)
        avg_test_accuracy = np.array(test_accuracy)

        # print(avg_test_accuracy)
        
        gtrl_mean = np.mean(avg_train_loss, axis=0)
        
        gtra_mean = np.mean(avg_train_accuracy, axis=0)
        # print(gtra_mean)
        gtsl_mean = np.mean(avg_test_loss, axis=0)
        gtsa_mean = np.mean(avg_test_accuracy, axis=0)

        gtrl_std = np.std(avg_train_loss, axis=0)
        gtra_std = np.std(avg_train_accuracy, axis=0)
        gtsl_std = np.std(avg_test_loss, axis=0)
        gtsa_std = np.std(avg_test_accuracy, axis=0)

        gtrl_mean_std = np.column_stack((gtrl_mean, gtrl_std))
        gtra_mean_std = np.column_stack((gtra_mean, gtra_std))
        gtsl_mean_std = np.column_stack((gtsl_mean, gtsl_std))
        gtsa_mean_std = np.column_stack((gtsa_mean, gtsa_std))

        training_loss_mean_std = gtrl_mean_std[gtrl_mean_std[:,0].argmin()]
        training_acc_mean_std = gtra_mean_std[gtra_mean_std[:,0].argmax()]
        val_loss_mean_std = gtsl_mean_std[gtsl_mean_std[:,0].argmin()]
        val_acc_mean_std = gtsa_mean_std[gtsa_mean_std[:,0].argmax()]

        
        with h5py.File(directory_name  + '{}.h5'.format(avg_file), 'w') as hf:
            hf.create_dataset('avg_training_loss', data=gtrl_mean)
            hf.create_dataset('avg_training_accuracy', data=gtra_mean)
            hf.create_dataset('avg_test_loss', data=gtsl_mean)
            hf.create_dataset('avg_test_accuracy', data=gtsa_mean)
            hf.close



        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
        print("Algorithm :",algorithm)
        print("Global training loss (mean/std) : (",training_loss_mean_std[0],"/",training_loss_mean_std[1],")")
        print("Global training accuracy (mean/std) : (",training_acc_mean_std[0],"/",training_acc_mean_std[1],")")
        print("Global test loss (mean/std) : (", val_loss_mean_std[0],"/", val_loss_mean_std[1],")")
        print("Global test accuracy (mean/std) : (",val_acc_mean_std[0],"/",val_acc_mean_std[1],")")
        print("+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-\n")
        

"""
Get the average of accuracy and loss
"""

# path = "/proj/sourasb-220503/codebase/FedFWplus/results/FedFW/MNIST/MCLR/time_varing_eta/time_varing_lambda/perf/10/"
# path = "/proj/sourasb-220503/codebase/FedFWplus/results/FedProx/MNIST/CNN/perf/3/"
# path = "/proj/sourasb-220503/codebase/FedFWplus/results/FedProx/SYNTHETIC/DNN/perf/3/"
# path = "/proj/sourasb-220503/codebase/FedFWplus/results/FedFW/MNIST/MCLR/constant_eta/constant_lambda/perf/10/"
# directory_name = "./results/convergence/MNIST/10/mclr/"
#average_result(path, directory_name, 'Fedfw', 'fedfw_tel')
# average_result(path, 'FedavgSGD', 'fedavg_sgd.h5')
# average_result(path, 'FedavgPGD', 'fedavg_pgd.h5')
# average_result(path, 'FedavgPSGD', 'fedavg_psgd.h5')
# average_result(path, 'Fedprox', 'fedprox.h5')
# average_result(path, 'Fedfw', 'fedfw.h5')



path = "/proj/sourasb-220503/codebase/FedFWplus/results/convergence/MNIST/10/mclr/"
acc_file = "accuracy_sota"
loss_file = "loss_sota"

convergence_analysis(path, acc_file, loss_file)
