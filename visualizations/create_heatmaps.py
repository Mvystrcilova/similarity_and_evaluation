import seaborn as sns, pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
data = pandas.read_csv('../max_results_updated', index_col=0, header=0, sep=';')

def create_heatmap():
    # scaled_data = data.values
    # scaler = preprocessing.MinMaxScaler()
    # scaled_data = scaler.fit_transform(scaled_data)
    # new_df = pandas.DataFrame(scaled_data)
    new_df = data.rank(axis=0, ascending=False)
    print(new_df)
    fig = plt.figure()
    r = sns.heatmap(new_df, cmap="RdYlGn_r", linecolor="white", linewidths=0.8, annot=data, cbar=False, square=False,
                    fmt=".4g", annot_kws={"size": 15})
    r.set_yticklabels(data.index, rotation=-15, fontsize=15)
    r.set_xticklabels(data.columns, rotation=0, fontsize=15)
    r.set_ylabel('')
    r.xaxis.tick_top()  # x axis on top
    r.xaxis.set_label_position('top')
    # r.set_title("evaluation metric")
    plt.tight_layout()
    fig.savefig('../graphs/paper_heatmap.png', dpi=700)


    plt.show()

# create_heatmap()

table_data = pandas.read_csv('../differences', sep=';', header=0, index_col=0)
def create_table():
    fig = plt.figure(figsize=(22, 20))
    table = plt.table(cellText=table_data.values.tolist(), rowLabels=table_data.index.tolist(),
              colLabels=table_data.columns, loc='center', colWidths=[0.03, 0.02, 0.02, 0.03, 0.03, 0.07],
                      cellLoc='center', colColours=['mistyrose','mistyrose','mistyrose','mistyrose','mistyrose','mistyrose'],
                      rowColours=['honeydew','honeydew','honeydew','honeydew','honeydew','honeydew'], rowLoc='right')
    plt.axis('off')
    plt.grid('off')
    plt.tight_layout()
    plt.savefig('../graphs/difference_table.png', dpi=700, bbox_inches="tight")
    plt.show()

create_table()