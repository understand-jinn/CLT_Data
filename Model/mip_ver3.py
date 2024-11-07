from docplex.mp.model import Model
import os
import csv
import pandas as pd
import geometric_center_ver1 as gm
import figure as fig
import time

def mip_solver():
    
    print('Start CPLEX MIP Solver')
    model = Model(name = 'IP model')
    
    # check result folder
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)
        
    new_data_path = os.path.join(input_data_path, f'Container_ex{ex_id}.csv')
    initial_data_path = os.path.join(input_data_path, f'Initial_state_ex{ex_id}.csv')
    
    new_data_df = pd.read_csv(new_data_path)
    initial_data_df = pd.read_csv(initial_data_path)
    
    # Parameters
    initial_cont_num = len(initial_data_df)
    new_cont_num = len(new_data_df)
    total_cont_num = initial_cont_num + new_cont_num
    dist_min = 0
    dist_max = stack_num + tier_num - 2
    
    
    # initial -> x-axis : 1 ~ stack_num
    # initial -> y-axis : 0 ~ tier_num -1
    initial_data_df['cont_type'] = 'initial'
    initial_data_df['seq'] = 0
    new_data_df['cont_type'] = 'new'
    new_data_df['loc_y'] = 0
    
    container_df = pd.concat([initial_data_df, new_data_df], axis=0).reset_index(drop=True)
    container_df['score'] = container_df['weight'] + (container_df['group'] * 100)
    
    initial_df = container_df[container_df['cont_type'] == 'initial']
    initial_cont_idx_list = initial_df.index.tolist()
    new_df = container_df[container_df['cont_type'] == 'new']
    new_cont_idx_list = new_df.index.tolist()
    
    geometric, container_df = gm.get_geometric_center(stack_num, tier_num, container_df, level_num)
 
    print('container_df : \n', container_df, '\n')
    
    # # Decision variables
    x = model.binary_var_dict([(i,j,k) for i in range(total_cont_num) for j in range(stack_num) for k in range(tier_num)], lb = 0, ub = 1, name = 'x')
    r = model.binary_var_dict([(j,k) for j in range(stack_num) for k in range(tier_num)], lb = 0, ub = 1, name = 'r')

    dist = model.continuous_var_dict([i for i in new_cont_idx_list], lb = 0, name = 'dist')
    dist_x = model.continuous_var_dict([i for i in new_cont_idx_list], lb = 0, name = 'dist_x_')
    dist_y = model.continuous_var_dict([i for i in new_cont_idx_list], lb = 0, name = 'dist_y_')
    # dist_norm = model.continuous_var_dict([i for i in new_cont_idx_list], lb = 0, name = 'dist_norm_')
    

    # Conaraints
    # 1. Each container is located at only one storage location
    for i in range(total_cont_num):
        model.add_constraint(sum(x[i,j,k] for j in range(stack_num) for k in range(tier_num)) == 1)
        
    # 2. Each storage location has only one container
    for j in range(stack_num):
        for k in range(tier_num):
            model.add_constraint(sum(x[i,j,k] for i in range(total_cont_num)) >= 1)
                
    # 3. The height of stack j must be less than or equal to tier_num
    for j in range(stack_num):
        model.add_constraint(sum(x[i,j,k] for k in range(tier_num) for i in range(total_cont_num)) <= tier_num)
        
    # 4. if slot k+1 is occupied, slot k must be occupied
    for j in range(stack_num):
        for k in range(tier_num - 1):
            model.add_constraint(sum(x[i,j,k] for i in range(total_cont_num)) >= sum(x[i,j,k+1] for i in range(total_cont_num)))
            
    # 5. define dist, dist_x, dist_y
    for i in new_cont_idx_list:
        model.add_constraint(dist[i] == dist_x[i] + dist_y[i])
        model.add_constraint(dist_x[i] == sum(j * x[i,j,k] for j in range(stack_num) for k in range(tier_num)) - container_df.loc[i, 'centroid_x'])
        model.add_constraint(dist_y[i] == sum(k * x[i,j,k] for j in range(stack_num) for k in range(tier_num)) - container_df.loc[i, 'centroid_y'])
        # model.add_constraint(dist_norm[i] == (dist[i] - dist_min) / (dist_max - dist_min))
        
    # 6. peak stack limit
    for j in range(stack_num -1):
        model.add_constraint(sum(x[i,j,k] for k in range(tier_num) for i in range(total_cont_num)) - sum(x[i,j+1,k] for k in range(tier_num) for i in range(total_cont_num)) <= peak_limit)
        model.add_constraint(sum(x[i,j,k] for k in range(tier_num) for i in range(total_cont_num)) - sum(x[i,j+1,k] for k in range(tier_num) for i in range(total_cont_num)) >= -peak_limit)
        
    # 7. define r_jk
    for j in range(stack_num):
        for k in range(tier_num -1):
            model.add_constraint((sum(x[i,j,k] * container_df.loc[i, 'score'] for i in range(total_cont_num)) - sum(x[i,j,k+1] * container_df.loc[i, 'score'] for i in range(total_cont_num))) / M <= M * (1- sum(x[i,j,k+1] for i in range(total_cont_num))) + r[j,k] )
            model.add_constraint(r[j,k] <= M * (1 - sum(x[i,j,k+1] for i in range(total_cont_num))) + r[j,k+1])
    
    for j in range(stack_num):
        for k in range(tier_num):
            model.add_constraint(r[j,k] <= sum(x[i,j,k] for i in range(total_cont_num)))
            
    # 8. sequence constraint
    for j in range(stack_num):
        for k in range(tier_num -1):
            model.add_constraint(sum(x[i,j,k] * container_df.loc[i, 'seq'] for i in range(total_cont_num)) <= M * (1 - sum(x[i,j,k+1] for i in range(total_cont_num))) + sum(x[i,j,k+1] * container_df.loc[i, 'seq'] for i in range(total_cont_num)))
    
    # 9. initial container location constraint
    for i in initial_cont_idx_list:
        model.add_constraint(x[i, container_df.loc[i, 'loc_x'] - 1, container_df.loc[i, 'loc_z']] == 1)        
            
    model.minimize(alpha * sum(r[j,k] for j in range(stack_num) for k in range(tier_num)) + beta * sum((dist[i] - dist_min) / (dist_max - dist_min) for i in new_cont_idx_list))
    
    start_time = time.time()
    
    # Solve the model
    status = model.solve()

    end_time = time.time()

    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    
    if status:
        print('We got a Optimal Solution')

        save_solution_path = f'{save_result_folder}/Solution_ex{ex_id}.txt'
        
        with open(save_solution_path, 'w') as f:
            f.write(f'Number of initial container : {initial_cont_num}\n')
            f.write(f'Number of new container : {new_cont_num}\n')
            f.write(f'Total number of containers : {total_cont_num}\n')
            f.write(f'alpha : {alpha}, beta : {beta}\n')
            f.write(f'Level_num : {level_num}\n')
            f.write(f'container_df :\n{container_df}\n')
            f.write(f'geometric :\n{geometric}\n')
            f.write("---------------------------------\n")
                   
            f.write(f"Repeat number : {ex_id}\n")
            f.write(f'Time taken to solve the MIP model : {elapsed_time} seconds\n')  
            f.write(f'min of distance : {dist_min}, max of distance : {dist_max}\n')
            f.write("---------------------------------\n")
            f.write(model.solution.to_string())
            
            for i in range(total_cont_num):
                for j in range(stack_num):
                    for k in range(tier_num):
                        if x[i,j,k].solution_value >= 0.99:        
                            container_df.loc[i, 'loc_x'] = j + 1
                            container_df.loc[i, 'loc_z'] = k
                            container_df.loc[i, 'reloc'] = r[j,k].solution_value
        
        print(f'Updated dataframe \n{container_df}')
        fig.draw_figure(stack_num, tier_num, container_df, save_result_path)

    
    else:
        print('Infeasible Solution')
        save_solution_path = f'{save_result_folder}/Infeasible_ex{ex_id}.txt'
        
        
        with open(save_solution_path, 'w') as f:
            f.write("Can't find feasible solution\n\n")
            
            f.write(f'Number of initial container : {initial_cont_num}\n')
            f.write(f'Number of new container : {new_cont_num}\n')
            f.write(f'Total number of containers : {total_cont_num}\n')
            f.write(f'Level_num : {level_num}\n')
            f.write(f'alpha : {alpha}, beta : {beta}\n')        
            f.write(f"Repeat number : {ex_id}\n")
            f.write(f'container_df :\n{container_df}\n')
            f.write(f'geometric :\n{geometric}\n')
            
            f.write(f'Time taken to solve the MIP model : {elapsed_time} seconds\n')  

            f.write(f'min of distance : {dist_min}, max of distance : {dist_max}\n')
            f.write("---------------------------------\n")
    
    
# def main():
    
#     # read all input folders in input_data_path
#     input_folder_list = os.listdir(input_data_path)
    
#     for input_folder_name in input_folder_list:
#         input_folder_path = os.path.join(input_data_path, input_folder_name)
        
#         # initial folders
#         initial_folder_list = os.listdir(input_folder_path)
        
#         for initial_folder_name in initial_folder_list:
            
#             initial_folder_path = os.path.join(input_folder_path, initial_folder_name)
            
#             # new folder
#             new_folder_list = os.listdir(initial_folder_path)
            
#             for new_folder_name in new_folder_list:
                
#                 new_folder_path = os.path.join(initial_folder_path, new_folder_name)
#                 print(input_folder_name)
#                 cont_stack_info = input_folder_name.split('Input_Data_')[1]
#                 output_folder_path = os.path.join(output_data_path, f'{initial_folder_name}_Output_Data_{cont_stack_info}')
                
#                 print(output_folder_path)
                


# Parameters
stack_num = 6
tier_num = 5
peak_limit = 2
level_num = 9
alpha_list = [0, 0.5, 1]

M = 100

ex_id = 14
initial_container_num = 0
total_container_num = 7
new_container_num = total_container_num - initial_container_num

input_folder_name = 'Input_Data_{}(stack_6_tier_5)'
input_data_path = f'Ungrouped_Input_Data_ver0/Input_Data_{total_container_num}(stack_{stack_num}_tier_{tier_num})/Initial_{initial_container_num}/New_{new_container_num}'
result_folder = 'Ungrouped_Data(mip_ver3)'

for alpha in alpha_list:
    beta = 1 - alpha
    save_result_folder = f'{result_folder}/Initial_{initial_container_num}_Output_Data_{total_container_num}(stack_{stack_num}_tier_{tier_num})/alpha_{alpha}_beta_{beta}'
    save_result_path = f'{save_result_folder}/Container_ex{ex_id}.png'
    mip_solver()
    
    
    

# input_data_path = 'C:/Users/USER\workspace/CLT_Data/Ungrouped_Input_Data_ver0'
# output_data_path = 'CP_Result'

