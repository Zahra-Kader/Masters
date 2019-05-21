import map_making_functions

def real_space_estimator(my_map, ideal_filter):
    my_map_filtered=map_making_functions.apply_filter(my_map, ideal_filter)
    product_map=my_map_filtered*my_map
    return product_map

def real_space_estimator2(my_map1,my_map2, ideal_filter):
    my_map_filtered=map_making_functions.apply_filter(my_map1, ideal_filter)
    product_map=my_map_filtered*my_map2
    return product_map

#real space estimators
def real_space_RS_estimator(my_map, ideal_RS_filter):
    my_map_filtered=map_making_functions.apply_RS_filter(my_map, ideal_RS_filter)
    nside=ideal_RS_filter.shape[0]
    diff=0 # nside/5 #0#nside/10 - CHECK THIS !
    my_map_cut=my_map[diff:nside-diff+1, diff:nside-diff+1]
    product_map=my_map_filtered*my_map_cut
    return product_map

def real_space_RS_estimator2(my_map1,my_map2, ideal_RS_filter):
    my_map_filtered=map_making_functions.apply_RS_filter(my_map1, ideal_RS_filter)
    
    nside=ideal_RS_filter.shape[0]
    diff=nside/8
    my_map2_cut=my_map2[diff:nside-diff+1, diff:nside-diff+1]
    product_map=my_map_filtered*my_map2_cut
    return product_map


