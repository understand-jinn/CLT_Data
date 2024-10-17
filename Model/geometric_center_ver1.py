import numpy as np
    
def div_level(_weights, _level_num):
    w_min = min(_weights)
    w_max = max(_weights)

    _level_range = []
    for l_c in range(1, _level_num +1):
        _level_range.append((w_min + (l_c -1) * ((w_max - w_min) / (_level_num - 1)), w_min + l_c * ((w_max - w_min) / (_level_num -1))))
    return _level_range


def get_level(_container_df, _level_range):    
    
    _container_df['level'] = 0
    
    for cont_idx in range(len(_container_df)):
        
        level_idx = 0
        
        for l_min, l_max in _level_range:
            
            level_idx += 1
            
            if _container_df['score'][cont_idx] >= l_min and _container_df['score'][cont_idx] < l_max:
                _container_df.loc[cont_idx,'level'] = level_idx
                
    return _container_df  

# 왼쪽에서 오른쪽으로 숫자 채우기
def set_geometric_grid(_stack_idx, _height_idx, min_stack_idx, max_stack_idx, min_height_idx, max_height_idx, sorted_levels, _grid, _set_container_num):
        
    # Set the next location
    if _set_container_num < len(sorted_levels):    
        
        _grid[_height_idx][_stack_idx] = sorted_levels[_set_container_num]
        _set_container_num += 1
        
        if _stack_idx == max_stack_idx:
            if min_height_idx != 0:
                min_height_idx -= 1

            #(0, 0) is full
            else:
                min_stack_idx += 1
                
            # stack_idx is max
            if max_stack_idx == _grid.shape[1] - 1:
                max_height_idx -= 1

            else: 
                max_stack_idx += 1
                
 
            _height_idx = min_height_idx
            _stack_idx = min_stack_idx
            
        else:
            if _stack_idx != max_stack_idx:
                _stack_idx += 1
                _height_idx += 1
        
        # print(grid, '\n')
        set_geometric_grid(_stack_idx, _height_idx, min_stack_idx, max_stack_idx, min_height_idx, max_height_idx, sorted_levels, _grid, _set_container_num)
    
    else:
        print('\nAll containers are placed')
        # print('geometric grid by level \n', _grid, '\n ------------------- \n')

    return _grid

# get dictionary of geometric center
def get_geometric_dict(_grid):
    _dict = {}

    for _h in range(_grid.shape[0]):
        for _s in range(_grid.shape[1]):
            # stack : 0 ~ stack_num, height : 0 ~ tier_num - 1 
            location = (_s, _grid.shape[0] - _h -1)
        
            if _grid[_h][_s] not in _dict.keys():
                # _dict[_grid[_h][_s]] = [(_h + 1, _s +1)]
                # print(_s, _h, _grid.shape[0], _grid.shape[1])
                _dict[_grid[_h][_s]] = [location]
                
            else:
                # append the same level
                # _dict[_grid[_h][_s]].append((_h +1, _s +1))
                _dict[_grid[_h][_s]].append(location)
                
    # print('best locations \n', _dict, '\n')
    return _dict



# # get geometric center
# def get_geometric_center(_m, _h, _weights, _level_num):
    
    level_range = div_level(_weights, _level_num)
    print('level range : ', level_range, '\n')
    
    container_level = get_level(_weights, level_range)
    
    grid = np.zeros((_h, _m))
    
    # sort weights in increasing order
    sorted_levels = sorted(container_level)
    
    # Height_idx : 0 ~ 4
    height_idx = _h - 1
    stack_idx = 0
    stack_max_idx = stack_idx
    stack_min_idx = stack_idx
    height_max_idx = height_idx
    height_min_idx = height_idx
    set_container_num = 0
    
    
    geometric_grid = set_geometric_grid(stack_idx, height_idx, stack_min_idx, stack_max_idx, height_min_idx, height_max_idx, sorted_levels, grid, set_container_num)
    print('geometric grid by level \n', geometric_grid, '\n ------------------- \n')

    
    geometric_dict = get_geometric_dict(geometric_grid)

    _geometric_center_dict = {}

    if len(_weights) == len(container_level):
        for c_l in container_level:
            x = 0
            y = 0
            for i in range(len(geometric_dict[c_l])):
                x += geometric_dict[c_l][i][0]
                y += geometric_dict[c_l][i][1]
            
            x_avg = x / len(geometric_dict[c_l])
            y_avg = y / len(geometric_dict[c_l])    
            _geometric_center_dict[c_l] = (x_avg, y_avg)
  
    else:
        print('There is a problem in dividing levels.')
        
    print('geometric_center : ', _geometric_center_dict, '\n')
    return geometric_grid, _geometric_center_dict, container_level

# get geometric center
def get_geometric_center(_m, _h, _container_df, _level_num):
    
    container_weights = _container_df['weight']
    
    level_range = div_level(container_weights, _level_num)

    _container_df = get_level(_container_df, level_range)
    
    grid = np.zeros((_h, _m))
    
    container_level = _container_df['level'].tolist()
    
    # sort weights in increasing order
    sorted_levels = sorted(container_level)
    
    # Height_idx : 0 ~ 4
    height_idx = _h - 1
    stack_idx = 0
    stack_max_idx = stack_idx
    stack_min_idx = stack_idx
    height_max_idx = height_idx
    height_min_idx = height_idx
    set_container_num = 0
    
    
    geometric_grid = set_geometric_grid(stack_idx, height_idx, stack_min_idx, stack_max_idx, height_min_idx, height_max_idx, sorted_levels, grid, set_container_num)
    print('geometric grid by level \n', geometric_grid, '\n ------------------- \n')

    
    geometric_dict = get_geometric_dict(geometric_grid)

    _geometric_center_dict = {}

    _container_df['centroid_x'] = 0
    _container_df['centroid_y'] = 0
    
    for c_l in container_level:
        x = 0
        y = 0
        for i in range(len(geometric_dict[c_l])):
            x += geometric_dict[c_l][i][0]
            y += geometric_dict[c_l][i][1]
        
        x_avg = x / len(geometric_dict[c_l])
        y_avg = y / len(geometric_dict[c_l])    
        _geometric_center_dict[c_l] = (x_avg, y_avg)
        
        container_level_df = _container_df['level'] == c_l
        _container_df.loc[container_level_df, 'centroid_x'] = x_avg
        _container_df.loc[container_level_df, 'centroid_y'] = y_avg
    
    return geometric_grid, _container_df
    # return geometric_grid, _geometric_center_dict, container_level

def get_scroe_list(weights, _group):
    
    scores = []
    
    for i in range(len(weights)):
        weight = weights[i] + (_group[i] * 100)
        scores.append(weight)
    return scores


# level_num = 9
# # Number of stacks
# m = 6

# # Capacity of tiers
# h = 5

# w = np.arange(1, 9)

# height_idx = h - 1
# stack_idx = 0
# stack_max_idx = stack_idx
# stack_min_idx = stack_idx
# height_max_idx = height_idx
# height_min_idx = height_idx
# set_container_num = 0
# grid = np.zeros((h, m))

# get_geometric_center(m, h, w, level_num)
# geometric_grid = set_geometric_grid(stack_idx, height_idx, stack_min_idx, stack_max_idx, height_min_idx, height_max_idx, w, grid, set_container_num)
# 