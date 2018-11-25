from csv import DictReader
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

def pca_results(full_dataset, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)



def parse_datadict():
    dd = open('Data_Dictionary.md', 'rt')
    reader = DictReader(dd)

    # dictionaries for parsed, normalized data columns
    column_name = {'key': [], 'ColumnName': []}
    column_description = {'key': [], 'ColNameDescription': []}
    value_map = {'key': [], 'value_key': [], 'ValueMaps': []}
    column_meta = {'key': [], 'ColumnShortName': [], 'ColumnMeta': []}

    def get_col_key(order_dict_row):
        row_value = list(order_dict_row.values())
        row_key = row_value[0]
        row_key = row_key[row_key.find(' ')+1:]
        row_key = row_key[:row_key.find(' ')]
        return re.sub(' ', '', row_key)[:-1]

    with open('Data_Dictionary.md', 'rt') as dd:
        reader = DictReader(dd)
        reader_list = list(reader)
        for row in reader_list:
            row_value = list(row.values())

            # extracting column names
            if row_value[0][:3] == '###' and row_value[0] != '### Table of Contents':
                row_key = row_value[0]
                row_key = row_key[row_key.find(' ')+1:]
                row_key = row_key[:row_key.find(' ')]
                row_key = re.sub(' ', '', row_key)[:-1]

                # extracting individual columns
                for item in row_value:
                    # finding lists
                    if type(item) == list:
                        # extracting columns from list
                        for sub_item in item:
                            # avoiding blanks
                            if sub_item != '':
                                # booya, extra columns                            
                                # clean column
                                sub_item = re.sub(' ', '', sub_item)   

                                column_name['key'].append(row_key)
                                column_name['ColumnName'].append(sub_item)
                                #print(sub_item)
                    else:
                        # columns not in list
                        item = re.sub('[#.]', '', item)[1:]
                        item = re.sub(' ', '', item[item.find(' ')+1:])

                        column_name['key'].append(row_key)
                        column_name['ColumnName'].append(item)
                        #print(item)

        # setting column dataframe     
        df_colname = pd.DataFrame(column_name)
        df_colname = df_colname.where(df_colname!='', np.nan)
        df_colname.key.ffill(inplace=True)



        # extracting column name descriptions  
        match_key = df_colname.key.unique()   

        i = 0
        skip_append_list = []
        for row in reader_list:

            row_key = get_col_key(row)

            j = 1
            j_append = 1
            if row_key in match_key:  
                #print(row_key)


                row_desc = (list(reader_list[i+j].values())[0])


                while re.sub('[A-Z _]', '', row_desc) == '###':
                    j += 1
                    row_desc = (list(reader_list[i+j].values())[0])   
                #print(row_desc)

                #  finds row description
                while True:
                    try:
                        row_key_while = get_col_key(reader_list[i+j])
                        row_desc = (list(reader_list[i+j].values())[0])   
                    except:
                        pass

                    # until it reaches another match key
                    if row_key_while in match_key: break   
                    # until it reaches a level 2 heading
                    if re.sub('[A-Za-z0-9. -]', '', row_desc) == '##': break
                    # until it reaches a row with dashes
                    if row_desc == '-----': break
                    # if else goes wrong avoid infinite loop
                    if j > 30: break

                    k = 1
                    row_full_desc = []
                    row_full_desc.append(row_desc)

                    # finds the next row description for parts of it in new lines
                    while True:
                        try:
                            next_row_desc = (list(reader_list[i+j+k].values())[0])
                        except:
                            pass

                        # until it reaches another row description or heading
                        if next_row_desc[:1] in ['-', '#']: break

                        # appends to the full description
                        skip_append_list.append(next_row_desc)
                        row_full_desc.append(next_row_desc)

                        k += 1

                        # if else goes wrong, avoid infinite loop
                        if k > 10: break

                    row_full_desc = ' '.join(row_full_desc)
                    if row_full_desc not in skip_append_list:
                        #print(row_key, j_append, row_full_desc)

                        # appending column_description
                        if j_append == 1:
                            column_description['key'].append(row_key)
                            column_description['ColNameDescription'].append(row_full_desc)
                        else:
                            #print(row_key, row_full_desc)

                            # separating value key from value description
                            loc_colon = row_full_desc.find(':')

                            # if not colon but semicolon typo
                            if loc_colon < 0:
                                loc_colon = row_full_desc.find(';')

                            value_key = row_desc[2:loc_colon]
                            value_name = row_desc[loc_colon+2:]

                            try:
                                int(value_key)
                                value_map['key'].append(row_key)
                                value_map['value_key'].append(value_key)
                                value_map['ValueMaps'].append(value_name)
                            except:
                                #print('key', row_key,'val_key:', value_key,'value:', value_name)
                                column_meta['key'].append(row_key)
                                column_meta['ColumnShortName'].append(value_key)
                                column_meta['ColumnMeta'].append(value_name)


                    j += 1 # increase row_description slicing
                    j_append += 1 # increase rows to append to column_description

            i += 1

        df_column_description = pd.DataFrame(column_description)
        df_value_map = pd.DataFrame(value_map)
        df_value_map.drop_duplicates(inplace=True)
        df_column_meta = pd.DataFrame(column_meta)
        
    return df_colname, df_column_description, df_value_map, df_column_meta 