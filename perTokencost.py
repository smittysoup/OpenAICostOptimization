import matplotlib.pyplot as plt
import numpy as np
import costData
import os
import Cost_compare_app as app


class CostComparisonGraph:
    
    
    def __init__(self, file_path, fine_tune_size, fine_tune_model, api_model, max_graph, graph_step):
        self.fine_tune_size = fine_tune_size
        self.fine_tune_model = fine_tune_model
        self.api_model = api_model
        self.max_graph = max_graph
        self.graph_step = graph_step
        self.file_path = file_path
        
        self.costpertunetoken = costData.finetune_per_token[self.fine_tune_model]
        self.costpertokentuned = costData.tuned_per_token[self.fine_tune_model]
        self.costpertokenapi = costData.api_per_token[self.api_model]

    def equation1(self, x):
        return (
            (self.costpertunetoken * self.fine_tune_size) / 100 + (self.costpertokentuned * x) / 100
        )

    def equation2(self, x):
        return (self.costpertokenapi * x) / 100

    def plot_graph(self):
        # Generate x values
        x = np.linspace(0, self.max_graph, int(self.graph_step))

        # Calculate y values for both equations
        y1 = self.equation1(x)
        y2 = self.equation2(x)

        # Plot the graph
        plt.plot(x, y1, label='Fine Tune ' + self.fine_tune_model)
        plt.plot(x, y2, label=self.api_model)
        plt.title('Cost Comparison: Fine Tune ' + self.fine_tune_model + ' Original vs. ' + self.api_model)
        plt.xlabel('Number of Tokens')
        plt.ylabel('Cost in $')
        plt.legend()
        plt.grid(True)
        
         # Save the graph image      
        
        plt.savefig(self.file_path)
        plt.close()          
"""
# Variables
fine_tune_size = 50000
fine_tune_model = 'Davinci'
api_model = 'GPT 3.5'
max_graph = 100000
graph_step = (max_graph / 10)


# Create an instance of CostComparisonGraph
graph = CostComparisonGraph(fine_tune_size, fine_tune_model, api_model, max_graph, graph_step)

# Plot the graph
graph.plot_graph()
"""
