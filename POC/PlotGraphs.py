import matplotlib.pyplot as plt
import numpy as np

class PlotGraphs:
    # plot comparison graph
    def plot_graph(self, labels, values):
        fig, ax = plt.subplots()
        bar_width = 0.5
        opacity = 0.9

        for ind in range(len(labels)):
            rect = ax.bar(ind, values[ind], bar_width, 
                            alpha=opacity,
                            label=labels[ind])
            self.__autolabel(ax, rect)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Times (in s)')
        ax.set_title('Times by Methods')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.legend()
        fig.tight_layout()

        plt.show()

    # Attach a text label above each bar displaying its height
    def __autolabel(self, ax, rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                    str(round(height, 3)),
                    ha='center', va='bottom')