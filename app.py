from flask import Flask, render_template
import pickle

app = Flask(__name__)

def load_simulation_data():
    # Load your pickle file
    pickle_file_path = 'stock_simulation_data_nestle.pkl'
    with open(pickle_file_path, 'rb') as pickle_file:
        simulation_data = pickle.load(pickle_file)

    return simulation_data

@app.route('/')
def index():
    simulation_results = load_simulation_data()
    return render_template('index.html', results=simulation_results)

if __name__ == '__main__':
    app.run(debug=True)
