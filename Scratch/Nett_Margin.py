import pandas as pd

def main(inp,demand): 
    #Result Profit
    Net = 0  
    residual_cost = 0
    
    #Read Input
    def read_data(inp):
        data =  pd.read_excel(inp)
        data.columns=['GUDANG','BIAYA_HANDLING','ONGKIR','INDEX_MARGIN', 'CAPACITY']
        return data
    
    #Nett_Margin function
    def Nett_Margin(INDEX, ONGKIR,HANDLING): 
        NET_MARGIN = INDEX-(ONGKIR+HANDLING) 
        return NET_MARGIN
    
    #Calculate Profit   
    data['MAX_PROFIT'] = data.apply(lambda row : Nett_Margin(row['INDEX_MARGIN'], row['ONGKIR'],
                                                            row['BIAYA_HANDLING']), axis = 1) 
    data_sort = data.sort_values(by=['MAX_PROFIT'], ascending = False).reset_index()
    del data_sort['index']
    temp = 0
    for i in range(len(data)):
        if data_sort['CAPACITY'].iloc[i] >= demand:
            Net = data_sort['MAX_PROFIT'][i]*demand
            print(Net)
            break
        else:
            diff = demand - data_sort['CAPACITY'].iloc[i] 
            Net += data_sort['MAX_PROFIT'].iloc[i]*data_sort['CAPACITY'].iloc[i]
            if data_sort['CAPACITY'].iloc[i] + diff == demand:
                Net += data_sort['MAX_PROFIT'].iloc[i+1]*diff
                print(Net)
                break
    return Net 

if __name__ == '__main__': 
    #input data
    data = read_data('Demand_Bdg_AA.xlsx') #example
    demand = 20000                         #example
    #Main
    main(data,demand)