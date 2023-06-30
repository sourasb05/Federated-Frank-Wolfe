import copy
from tqdm import trange
from src.utils.utils import select_users

class FedFW_Server():
    def __init__(self, aggregator_name, model, global_iters, eta_t):
        self.x_bar_t = copy.deepcopy(model[0])
        # self.s_it_agg = copy.deepcopy(model[0])
        self.aggregator_name = aggregator_name
        self.global_iters = global_iters
        self.eta_t = eta_t
        
    def global_update(self, users, selected_users): 
        for param in self.x_bar_t.parameters():
            param.data.zero_()
        for user in selected_users:
            for x_bar_t_param, x_it_param in zip(self.x_bar_t.parameters(), user.x_it.parameters()):
                x_bar_t_param.data += x_it_param.data*(1/len(selected_users))
        
                
    def send_parameters(self, users):   
        if len(users) == 0:
            assert(f"AssertionError : The client can't be zero")
        else:
            for user in users:
                user.set_parameters(self.x_bar_t)

    
    def train(self, users):
        for iter in trange(self.global_iters):
            
            self.send_parameters(users)   # server send parameters to every users
            self.evaluate(users)  # evaluate global model
            selected_users = select_users(users)
            print("number of selected users",len(selected_users))
            for user in selected_users:
                user.local_train(self.x_bar_t)
            
            self.global_update(users, selected_users)

                
    def evaluate(self, users):
        tot_train_loss = 0
        tot_test_loss = 0
        tot_train_corrects = 0
        tot_test_corrects = 0
        tot_samples_train = 0
        tot_samples_test = 0
        
        """
            train_stats[0] = training accuracy
            train_stats[1] = number of training sample y.shape[0]
            train_stats[2] = training loss
            

            test_stats[0] = test accuracy
            test_stats[1] = y.shape[0]
            test_stats[2] = test loss
        """

        for user in users:
            train_stats = user.global_eval_train_data(self.x_bar_t)
            test_stats = user.global_eval_test_data(self.x_bar_t)

            tot_train_corrects += train_stats[0]
            tot_samples_train += train_stats[1]
            tot_train_loss += train_stats[2]

            
            tot_test_corrects += test_stats[0]
            tot_samples_test += test_stats[1]
            tot_test_loss += test_stats[2]


        avg_train_loss = tot_train_loss/len(users)
        avg_test_loss = tot_test_loss/len(users)
        avg_train_accuracy = tot_train_corrects/tot_samples_train
        avg_test_accuracy = tot_test_corrects/tot_samples_test

        print("Average train loss :", avg_train_loss.item())
        print("Average test loss :", avg_test_loss.item())
        print("Average train accuracy :", avg_train_accuracy)
        print("Average test accuracy :", avg_test_accuracy)