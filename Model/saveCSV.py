import csv
import os
import random
import numpy as np


def get_random_location(container_loc_list, weight_list, now_weight_idx, available_stack_list, tier_num, initial_state, stack_state):
    
    initial_container_num = len(weight_list)
    processed_container_num = len(container_loc_list)
    print('processed_container_num : ', processed_container_num, 'initial_container_num : ', initial_container_num, '\n')
    
    if processed_container_num < initial_container_num:
        
        weight = weight_list[now_weight_idx] 
        
        # choose random stack
        stack_idx = random.choice(available_stack_list)
        
        print('stack_idx : ', stack_idx, '\n')

        # 해당 Stack에 Container가 없을 경우
        if stack_state[stack_idx] == 0:
            tier_idx = tier_num - 1
            print('There is no container in stack : ', stack_idx, '\n')
            print('tier idx : ', tier_idx, ', stack idx : ', stack_idx, '\n')
            
            initial_state[tier_idx][stack_idx] = weight
            container_loc_list.append((stack_idx + 1, tier_num - tier_idx - 1))
            
            processed_container_num += 1
            stack_state[stack_idx] += 1
            
            print(initial_state, '\n')
            
        
        # 해당 Stack에 Container가 있을 경우
        else:
            if stack_state[stack_idx] < tier_num:
                tier_idx = tier_num - 1 - stack_state[stack_idx]
                initial_state[tier_idx][stack_idx] = weight
                
                container_loc_list.append((stack_idx + 1, tier_num - tier_idx - 1))
                
                processed_container_num += 1
                stack_state[stack_idx] += 1
                
                # print(initial_state, '\n')
                
                if stack_state[stack_idx] == tier_num:
                    # remove stack_idx that is index of available_stack_list
                    available_stack_list.remove(stack_idx)             
        print('-----------------------------------------\n')  
        
        now_weight_idx += 1
        get_random_location(container_loc_list, weight_list, now_weight_idx, available_stack_list, tier_num, initial_state, stack_state)
    
    
    else:
        print('All initial containers are placed')
        print('Initial State : \n', initial_state, '\n')
        print('Location of containers : ', container_loc_list, '\n')
    
    return container_loc_list, initial_state
    




    
# Create CSV file
def InitialContainerCSV(folderpath, fileName, ex_idx, start_idx, initial_container_num, stack_num, tier_num, group_list):
    
    # Column Name : idx,loc_x,loc_y,loc_z,weight,size(ft)  
    
    loc_y_list = [0 for _ in range(initial_container_num)]

    # weight_list = [round(random.uniform(5.0, 24.0), 2) for i in range(initial_container_num)]
    weight_list = [round(random.uniform(5.0, 24.0)) for i in range(initial_container_num)]

    size_list = [20 for _ in range(initial_container_num)]
    
    weight_idx = 0
    
    sorted_weight = np.sort(weight_list)
    # sort the weight in increasing order
    print('sorted weight : ', sorted_weight, '\n')
    
    # Create zero two-dimensional list
    initial_status = np.zeros((tier_num, stack_num))
    available_stack = [i for i in range(stack_num)]
    
    # number of containers in each stack
    stack_status = [0 for _ in range(stack_num)]
    # available stack from 0 to stack_num - 1
    container_locations = []
    
    container_locations, _initial_state = get_random_location(container_locations, sorted_weight, weight_idx, available_stack, tier_num, initial_status, stack_status)
    
    # save to text file
    Initial_state_file_path = os.path.join(folderpath, f'Initial_state_ex{ex_idx}.txt')
    
    with open(Initial_state_file_path, 'w') as f:
        f.write(f'Initial state : \n{_initial_state}')
            

    with open(os.path.join(folderpath, fileName + '.csv'), 'w', newline='') as csvfile:
        fieldnames = ['idx', 'loc_x', 'loc_y', 'loc_z', 'weight', 'group', 'size(ft)']
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(weight_list)):
            idx = start_idx + i
            loc_x = container_locations[i][0]
            loc_y = loc_y_list[i]
            loc_z = container_locations[i][1]
            weight = weight_list[i]
            group = group_list[i]
            size = size_list[i]
            
            # Write row to CSV file
            writer.writerow({'idx': idx, 'loc_x': loc_x, 'loc_y': loc_y, 'loc_z': loc_z, 'weight': weight, 'group' : group, 'size(ft)': size})
        
        print('--------- Success Create Input Data : ', fileName ,'---------\n')


# Create CSV file
def NewContainerCSV(folderpath, fileName, start_idx, new_container_num, group_list):
    
    # Column Name : idx,seq,group,weight,size(ft)
    
    # random sequence : 1 ~ new_con_num 
    # sequence_list = random.sample(range(1, new_container_num + 1), new_container_num)
    sequence_list = np.arange(1, new_container_num + 1)

    # random weight from 5.0 to 24.0 with up to 2 decimal places
    # weight_list = [round(random.uniform(5.0, 24.0), 2) for i in range(new_container_num)]
    weight_list = [round(random.uniform(5.0, 24.0)) for i in range(new_container_num)]
    
    
    size_list = [20 for _ in range(new_container_num)]
    
    with open(os.path.join(folderpath, fileName + '.csv'), 'w', newline='') as csvfile:
        fieldnames = ['idx', 'seq', 'group', 'weight', 'size(ft)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(sequence_list)):
            idx = start_idx + i
            seq = sequence_list[i]
            group = group_list[i]
            weight = weight_list[i]
            size = size_list[i]
        
            # Write row to CSV file
            writer.writerow({'idx': idx, 'seq': seq, 'group': group, 'weight': weight, 'size(ft)': size})
        
        print('--------- Success Create Input Data : ', fileName ,'---------\n')