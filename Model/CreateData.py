import saveCSV
import os
import random


stack_num = 6
tier_num = 5
container_num = 15
repeat_num = 30

initial_con_num_list = [0, 5, 7, 10, 15]
new_con_num_list = []
for _initial_num in initial_con_num_list:
    new_con_num_list.append(container_num - _initial_num)

initial_con_start_idx = 1

folder_name = f'Input_Data_{container_num}(stack_{stack_num}_tier_{tier_num})'

def get_priority(_group_list, container_num):
    
    container_group = []
    
    for i in range(container_num):
        # random choice in _priority_list
        group = random.choice(_group_list)
        
        # if priority != 0:
        #     # remove priority from _priority_list
        #     _priority_list.remove(priority)
        
        container_group.append(group)
        
    # return container_priority, _priority_list
    return container_group
       

def get_random_data(repeat_num, initial_con_num_list, new_con_num_list, stack_num, tier_num, initial_container_start_idx):
    if len(initial_con_num_list) ==  0 or len(new_con_num_list) == 0:
        print('Error : Check initial_con_num or new_con_num_list')
    else:    
        for container_num_idx in range(len(initial_con_num_list)):
            
            initial_con_num = initial_con_num_list[container_num_idx]
            new_con_num = new_con_num_list[container_num_idx]
            
            new_con_start_idx = initial_container_start_idx + initial_con_num
            # folderPath = 'Sample/Initial_' + str(initial_con_num) + '/New_' + str(new_con_num)
            folderPath = os.path.join(folder_name, f'Initial_{initial_con_num}', f'New_{new_con_num}')
            print('Folder Path : ', folderPath, '\n')
            
            # Check exist folder
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
                print('Create Folder : ', folderPath, '\n')
    
            for repeat_num_idx in range(repeat_num):
                
                initial_container_name = 'Initial_state_ex' + str(repeat_num_idx + 1)
                new_container_name = 'Container_ex' + str(repeat_num_idx + 1)
                
                print('--------- Start Create Input Data ---------')
                print('Initial Container Number : ', initial_con_num, '\nNew Container Number : ', new_con_num)

                # group_num = 3
                # 0 ~ group_num
                # priority_list = [i for i in range(0, group_num+1)]
                                    
                # initial_con_priority = get_priority(priority_list, initial_con_num)
                initial_con_group = [0 for _ in range(initial_con_num)]
                
                # Save Input Data for Initial Container
                saveCSV.InitialContainerCSV(folderPath, initial_container_name, initial_container_start_idx, initial_con_num, stack_num, tier_num, initial_con_group)
                
                # new_con_priority = get_priority(priority_list, new_con_num)
                new_con_group = [0 for _ in range(new_con_num)]
                # Save Input Data for New Container
                saveCSV.NewContainerCSV(folderPath, new_container_name, new_con_start_idx, new_con_num, new_con_group)   
                
            # for new_con_num in new_con_num_list:
                
            #     new_con_start_idx = initial_container_start_idx + initial_con_num
            #     folderPath = 'Data/Initial_' + str(initial_con_num) + '/New_' + str(new_con_num) + '/Input'
            #     print('Folder Path : ', folderPath, '\n')
                
            #     # Check exist folder
            #     if not os.path.exists(folderPath):
            #         os.makedirs(folderPath)
            #         print('Create Folder : ', folderPath, '\n')
        
            #     for repeat_num_idx in range(repeat_num):
                    
            #         initial_container_name = 'Initial_state_ex' + str(repeat_num_idx + 1)
            #         new_container_name = 'Container_ex' + str(repeat_num_idx + 1)
                    
            #         print('--------- Start Create Input Data ---------')
            #         print('Initial Container Number : ', initial_con_num, '\nNew Container Number : ', new_con_num)

            #         # group_num = 3
            #         # 0 ~ group_num
            #         # priority_list = [i for i in range(0, group_num+1)]
                                      
            #         # initial_con_priority = get_priority(priority_list, initial_con_num)
            #         initial_con_priority = [0 for _ in range(initial_con_num)]
                    
            #         # Save Input Data for Initial Container
            #         saveCSV.InitialContainerCSV(folderPath, initial_container_name, initial_container_start_idx, initial_con_num, stack_num, tier_num, initial_con_priority)
                    
            #         # new_con_priority = get_priority(priority_list, new_con_num)
            #         new_con_priority = [0 for _ in range(new_con_num)]
            #         # Save Input Data for New Container
            #         saveCSV.NewContainerCSV(folderPath, new_container_name, new_con_start_idx, new_con_num, new_con_priority)


get_random_data(repeat_num, initial_con_num_list, new_con_num_list, stack_num, tier_num, initial_con_start_idx)
