#         try:
#             # pylint: disable=g-import-not-at-top
#             tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
#             plot_only = 50
#             low_dim_embs = tsne.fit_transform(patient_embeddings[:plot_only])
#             # dict_med = {}
#             # dict_med['UNK'] = 'unknown words'
#             # data = xlrd.open_workbook(r'C:\Users\win\Desktop\test_data_med\id_med_dict.xlsx')
#             # table = data.sheets()[0]
#             # nrows = table.nrows
#             # for row_num in range(nrows):
#             #     dict_med[str(row_num)] = table.row_values(row_num)[1]
#             #
#             # labels = [dict_med[reverse_dictionary[i]] for i in range(plot_only)]
#             labels = true_label[:plot_only]
#             plot_with_labels(low_dim_embs, labels, os.path.join(os.getcwd(), 'med2rep.png'))
        
#         except ImportError as ex:
#             print('Please install sklearn, matplotlib, and scipy to show embeddings.')
#             print(ex)
