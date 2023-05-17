import matplotlib.pyplot as plt
import numpy as np
import costData


class QueryComparisonGraph:
    def __init__(self, filepath, qadjust, avg_query_len, fine_tune_size, fine_tune_model, api_model, max_graph, graph_step):
        self.qadjust = qadjust
        self.filepath=filepath
        self.avg_query_len = avg_query_len
        self.total_len = self.qadjust + self.avg_query_len
        self.fine_tune_size = fine_tune_size
        self.fine_tune_model = fine_tune_model
        self.api_model = api_model
        self.max_graph = max_graph
        self.graph_step = graph_step
        
        self.costpertunetoken = costData.finetune_per_token[self.fine_tune_model]
        self.costpertokentuned = costData.tuned_per_token[self.fine_tune_model]
        self.costpertokenapi = costData.api_per_token[self.api_model]

    def equation1(self, x):
        finetune_per_query_cost = ((self.costpertunetoken * self.avg_query_len) / 100)
        finetune_startup = ((self.costpertokentuned) * self.fine_tune_size) / 100
        return finetune_startup + finetune_per_query_cost * x

    def equation2(self, x):
        return self.costpertokenapi * x

    def plot_graph(self):
        # Generate x values
        x = np.linspace(0, self.max_graph, int(self.graph_step))

        # Calculate y values for both equations
        y1 = self.equation1(x)
        y2 = self.equation2(x)

        # Plot the graph
        plt.plot(x, y1, label=f'Fine Tune {self.fine_tune_model}')
        plt.plot(x, y2, label=self.api_model)
        plt.title(f'Cost Comparison: Fine Tune {self.fine_tune_model} Original vs. {self.api_model}')
        plt.xlabel('Number of Queries')
        plt.ylabel('Cost in $')
        plt.legend()
        plt.grid(True)
        
        
        plt.savefig(self.filepath)
        plt.close() 

    
'''
# Variables
qadjust = 900
avg_query_len = 300
fine_tune_size = 50000
fine_tune_model = 'Davinci'
api_model = 'GPT 3.5'
max_graph = 100000
graph_step = (max_graph / 10)

# Create an instance of CostComparisonGraph
graph = QueryComparisonGraph(qadjust, avg_query_len, fine_tune_size, fine_tune_model, api_model, max_graph, graph_step)

# Plot the graph
graph.plot_graph()
'''
