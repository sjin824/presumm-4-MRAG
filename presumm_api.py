from flask import Flask, request, jsonify
from nltk.tokenize import sent_tokenize

# 可能有问题
from presumm import PreSumm
from presumm import train

app = Flask(__name__)

def sentence_ranking_by_presumm(configs, fulltext):
    return train.main(configs, fulltext)

@app.route('/presumm', methods=['POST'])
def api_produce_sentences():
    """
    API Endpoint: Select candidate sentences based on fulltext and return JSON.
    """
    # configs在这里预定义。如果需要再挪出去一层。这些config要大改
    configs = {
        'task': 'ext',
        'mode': 'test_text',
        'test_from': 'bertext_cnndm_transformer.pt',
        'result_path': 'logs/ext',
        'alpha': 0.95,
        'log_file': 'logs/ext/log.txt',
        'visible_gpus': '0'
    }
    try:
        # Parse input JSON
        data = request.get_json()
        fulltext = data.get("fulltext", "")
        if not fulltext:
            return jsonify({"error": "No input text provided."}), 400
        
        # zhenyun的不知道哪来的逻辑。考虑是否需要
        if fulltext[0] in ["“", "'", "”"] and fulltext[-1] in ["“", "'", "”"]:
            fulltext = fulltext[1:-1]

        sent_with_score = sentence_ranking_by_presumm(configs, fulltext)

        #[idx]有没有真实用处还未确定 注意这个 fulltext['sents_id_selected_by_bertsum'] = sent_with_score[idx][2]
        ids_by_presumm = sent_with_score[2]
        sentences_by_presumm = sent_with_score[0]
        sentences_with_scores_by_presumm = sent_with_score[1].tolist()
        sentence_order_by_presumm = sent_with_score[4]
        sentence_texts_order_by_presumm = sent_with_score[5]
        # 不知道zhenyun问什么又要做一遍tokenization。考虑是否需要
        sentences = sent_tokenize(fulltext)

        # Prepare JSON response
        response = {
            "ids_by_presumm" : ids_by_presumm,
            "sentences_by_presumm": sentences_by_presumm,
            "sentences_with_scores_by_presumm": sentences_with_scores_by_presumm,
            "sentence_order_by_presumm": sentence_order_by_presumm,
            "sentence_texts_order_by_presumm": sentence_texts_order_by_presumm,
            "sentences": sentences
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    # Run the Flask API server
    app.run(host="0.0.0.0", port=5001, debug=True)