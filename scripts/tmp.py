from sklearn.metrics import confusion_matrix

def plot_multi_confusion_matrix(df, y_true, dataset, save=False,
                                class_names=['damage_negative', 'damage_positive']):
    '''take a dataframe with predictors and y_true value, output multiple confusion matrix plot
    '''

    # 'M-CAP_rankscore', 0.4815 is 0.025 cutoff 0.642 is 0.05 cutoff
    col_dict = {'cadd>15': ('CADD_phred', 15), 'cadd>20': ('CADD_phred', 20),
                'eigen_pred>10': ('Eigen-phred', 10), 'eigen_pred>15': ('Eigen-phred', 15),
                'eigen_pc_pred>10': ('Eigen-PC-phred', 10),
                'MetaSVM>0': ('MetaSVM_rankscore', 0.82271), 'MetaLR>0': ('MetaLR_rankscore', 0.81122),
                'M_CAP>0.025': ('M-CAP_rankscore', 0.4815), 'PP2-HVAR': ('Polyphen2_HVAR_rankscore', 0.6280),
                'FATHM': ('FATHMM_converted_rankscore', 0.8235),
                'cnn_0.5': ('cnn_prob', 0.5), 'cnn_0.56': ('cnn_prob', 0.56)}

    y_preds, y_algos = [], []
    for key, (col, threshold) in col_dict.items():
        y_algos.append(key)
        y_preds.append(convert2binary(df, col, threshold))

    infos = []
    for y_pred, y_algo in zip(y_preds, y_algos):
        # Compute confusion matrix
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        cnf_matrix = confusion_matrix(y_true, y_pred)
        fdr = 1 - precision_score(y_true, y_pred)
        np.set_printoptions(precision=2)

        title = '../figure/' + dataset + y_algo + '.png'
        # Plot non-normalized confusion matrix
        figure_title = 'Confusion matrix, without normalization\n{}\n{}\n accuracy: {:.2f}\n f1: {:.2f}\n'.format(
            dataset, y_algo, accuracy, f1)
        fig = plt.figure(figsize=(5, 5))
        plot_confusion_matrix(cnf_matrix, classes=class_names,
                              title=figure_title)

        if save:
            infos.append([y_algo, accuracy, f1, fdr])
            fig.savefig(title)
            plt.close()
        else:
            plt.show()
    labels = ['Col', 'accuracy', 'f1', 'FDR']
    df = pd.DataFrame(infos, columns=labels)
    display(df)
