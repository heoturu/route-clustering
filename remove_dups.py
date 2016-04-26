def remove_dups(routes_coord):
    routes_coord_no2cycles_intl = []
    
    removed_route_count = 0
    for i in range(len(routes_coord)):
#         print(*routes_coord[i], sep='\n')
#         print()
        
        route_unique = list(map(itemgetter(0), groupby(routes_coord[i])))
        
#         print(*route_unique, sep='\n')
        
        final_route = []
        
        j = 0
        while j < len(route_unique):
#             init_j = j
#             print(j)
#             if route_unique[j] == route_unique[j + 2]

            if j > len(route_unique) - 1 - 4:
                final_route.append(route_unique[j])
                j += 1
            elif (route_unique[j] == route_unique[j + 2] and
                route_unique[j + 1] == route_unique[j + 3] and
                not route_unique[j] == route_unique[j + 4]):
                final_route.append(route_unique[j])
                j += 3
            elif (route_unique[j] == route_unique[j + 2] == route_unique[j + 4] and
                  route_unique[j + 1] == route_unique[j + 3]):
                k = j + 1
                while k < len(route_unique):
                    if (route_unique[k] != route_unique[j] and 
                        route_unique[k] != route_unique[j + 1]):
                        if route_unique[k - 1] == route_unique[j]:
                            final_route += [route_unique[j], route_unique[j + 1]]
                        else:
                            # I.e. route_unique[k - 1] == route_unique[j + 1]
                            final_route.append(route_unique[j])
                        j = k - 1
                        break
                    k += 1
                if k == len(route_unique):
                    if route_unique[len(route_unique) - 1] == route_unique[j]:
                        final_route += [route_unique[j], route_unique[j + 1], route_unique[j]]
                    else:
                        final_route += [route_unique[j], route_unique[j + 1]]
                    break
            else:
                final_route.append(route_unique[j])
                j += 1

#             if init_j == 1:
#                 print(j)
#                 raise Exception('Exc')
        
        if not len(final_route) == len(route_unique):
            print(i)
        else:
            print('not ' + str(i))
        
        routes_coord_no2cycles_intl.append(final_route)
    return routes_coord_no2cycles_intl
