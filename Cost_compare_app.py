from flask import Flask, render_template, request, url_for
import perTokencost
import perQuerycost
import os

if not os.path.exists('static'):
    os.makedirs('static')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    qadjust = int(request.form['qadjust'])
    avg_query_len = int(request.form['avg_query_len'])
    fine_tune_size = int(request.form['fine_tune_size'])
    fine_tune_model = request.form['fine_tune_model']
    api_model = request.form['api_model']
    max_graph1 = int(request.form['max_graph'])
    max_graph2 = int(request.form['max_graph2'])
    graph_step1 = max_graph1 / 10
    graph_step2 = max_graph2 / 10
    
    file_path = os.path.join(app.root_path, 'static', 'tokengraph.png')
    file_path2 = os.path.join(app.root_path, 'static', 'querygraph.png')
    graph1 = perTokencost.CostComparisonGraph(file_path, fine_tune_size, fine_tune_model, api_model, max_graph1, graph_step1)   
    graph1.plot_graph()
    graph2 = perQuerycost.QueryComparisonGraph(file_path2,qadjust, avg_query_len, fine_tune_size, fine_tune_model, api_model, max_graph2, graph_step2)
    graph2.plot_graph()
    image1_path = url_for('static', filename='tokengraph.png')
    image2_path = url_for('static', filename='querygraph.png')

    return render_template('index.html', image1=image1_path, image2=image2_path)

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)
