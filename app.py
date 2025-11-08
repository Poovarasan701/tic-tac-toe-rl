from flask import Flask, request, jsonify, render_template
import pickle
from ai import TicTacToeAI

app = Flask(__name__)

with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

ai = TicTacToeAI()
ai.q_table = q_table

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    board = data['board']
    move = ai.best_move(board)
    return jsonify({'move': move})

if __name__ == "__main__":
    app.run(debug=True)
